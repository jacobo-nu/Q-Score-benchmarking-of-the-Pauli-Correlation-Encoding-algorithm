"""
qscore_PCE.py
=============
Versión del benchmark Q-Score que sustituye QAOA por PCE (Pauli Correlation
Encoding) como algoritmo cuántico para resolver Max-Cut.

Reutiliza de qscore.py: la generación de grafos Erdős-Rényi, las clases
QScoreConfiguration / BenchmarkRunResult / BenchmarkAnalysisResult, la
función is_successful() y la métrica beta (approximation ratio).

Sustituye: el ansatz QAOA y compute_expectation_value/run_QAOA por la
maquinaria PCE (Circuit HEA, loss_func_estimator, build_sign_tensor, etc.)
adaptada del repositorio CESGA-Quantum-Spain-PCE-Benchmark.

PRIMERA VERSIÓN — simulación exacta (statevector), sin ruido, sin shots,
pensada solo para validar que el pipeline corre de principio a fin.
"""

import logging
import math
from dataclasses import dataclass
from itertools import combinations
from math import comb
from time import strftime
from typing import Dict, List, Tuple

import networkx as nx
from networkx import Graph
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize, differential_evolution
import xarray as xr

# ---------------------------------------------------------------------------
# Reutilizado de qscore.py
# ---------------------------------------------------------------------------

qcvv_logger = logging.getLogger("qscore_pce")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class QScoreConfiguration:
    """Configuración del benchmark Q-Score-PCE."""
    num_instances: int = 10
    min_num_nodes: int = 2
    max_num_nodes: int = 5
    node_step: int = 1              # incremento entre tamaños de grafo
    seed: int = 42

    # --- Parámetros específicos de PCE ---
    k: int = 2                      # orden de compresión PCE (2 = cuadrático)
    pce_maxiter: int = 10           # iteraciones del optimizador clásico
    pce_optimizer: str = "COBYLA"   # optimizador scipy ("COBYLA", "Powell", "BFGS"...)
    pce_alpha_factor: float = 1.5   # alpha = pce_alpha_factor * num_qubits
    pce_beta: float = 0.5           # beta de la regularización

    # --- Parámetros específicos de differential_evolution ---
    pce_de_popsize: int = 3
    pce_de_strategy: str = "best1exp"
    pce_de_mutation: tuple = (0.5, 1)
    pce_de_recombination: float = 0.7
    pce_de_tol: float = 1e-7
    pce_de_polish: bool = True
    pce_de_init: str = "halton"

    def __iter__(self):
        for k_, v in self.__dict__.items():
            yield k_, v


@dataclass
class BenchmarkRunResult:
    dataset: xr.Dataset


@dataclass
class BenchmarkObservation:
    name: str
    value: object
    uncertainty: float = 0.0


@dataclass
class BenchmarkAnalysisResult:
    dataset: xr.Dataset
    plots: Dict
    observations: List[BenchmarkObservation]


def is_successful(approximation_ratio: float) -> bool:
    """Mismo criterio que QAOA: beta* = 0.2."""
    return bool(approximation_ratio > 0.2)


def cut_cost_function(x: str, graph: Graph) -> int:
    """Número de aristas cortadas (con signo negativo, igual que en qscore.py)."""
    obj = 0
    for i, j in graph.edges():
        if x[i] != x[j]:
            obj += 1
    return -1 * obj


# ---------------------------------------------------------------------------
# Adaptado de auxiliar.py (CESGA-Quantum-Spain-PCE-Benchmark)
# ---------------------------------------------------------------------------

def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    discriminant = b ** 2 - 4 * a * c
    if discriminant >= 0:
        x_1 = (-b + math.sqrt(discriminant)) / (2 * a)
        x_2 = (-b - math.sqrt(discriminant)) / (2 * a)
    else:
        x_1 = complex((-b / (2 * a)), math.sqrt(-discriminant) / (2 * a))
        x_2 = complex((-b / (2 * a)), -math.sqrt(-discriminant) / (2 * a))
    return x_1, x_2


def num_qubits(num_variables: int, order_compression: int) -> int:
    """
    Número de qubits PCE necesarios para num_variables variables, orden k.

    select_nodes_from_aux reparte las m variables en tres bases:
    list_size = m // 3 valores de X, list_size de Y, y
    rem = m - 2*list_size de Z. Cada base solo dispone de
    comb(qubits, k) valores, así que hace falta garantizar
    comb(qubits, k) >= rem (el mayor de los tres bloques),
    no solo 3*comb(qubits, k) >= m como suma global.
    """
    if order_compression != 2:
        raise NotImplementedError("Esta primera versión solo soporta k=2.")

    qubits = math.ceil(max(solve_quadratic(1, -1, -2 / 3 * num_variables)))
    qubits = max(qubits, 2)

    # Verificación/ajuste de seguridad: comb(qubits, k) debe cubrir
    # el bloque más grande (rem, la base Z) que pide select_nodes_from_aux.
    list_size = num_variables // 3
    rem = num_variables - 2 * list_size
    while comb(qubits, order_compression) < rem:
        qubits += 1

    return qubits


def calc_cut_size(graph: Graph, partition0, partition1) -> float:
    """Igual que op_graph.calc_cut_size, nodos indexados desde 1."""
    cut_size = 0.0
    for edge0, edge1, data in graph.edges(data=True):
        if (edge0 in partition0 and edge1 in partition1) or \
           (edge0 in partition1 and edge1 in partition0):
            cut_size += data.get("weight", 1.0)
    return cut_size


def get_partition_from_expmap(exp_map: Dict[int, float], G: Graph):
    """A partir del mapa de expectativas, separa nodos en dos particiones."""
    par0 = {i + 1 for i, val in exp_map.items() if val >= 0}
    par1 = {i + 1 for i, val in exp_map.items() if val < 0}
    cut_size = calc_cut_size(G, par0, par1)
    bitstring_init = [1 if val >= 0 else 0 for val in exp_map.values()]
    return par0, par1, cut_size, bitstring_init


def local_refinement_from_partition(G: Graph, par0, par1):
    """Búsqueda local 1-swap para mejorar el corte (igual que el repo PCE)."""
    cur_bits = [1 if (i + 1) in par0 else 0 for i in range(len(G.nodes()))]
    best_bits = cur_bits.copy()
    best_cut = calc_cut_size(G, par0, par1)

    for edge0, edge1 in G.edges():
        swapped_bits = cur_bits.copy()
        swapped_bits[edge0 - 1], swapped_bits[edge1 - 1] = swapped_bits[edge1 - 1], swapped_bits[edge0 - 1]
        part0 = {i + 1 for i, b in enumerate(swapped_bits) if b}
        part1 = {i + 1 for i, b in enumerate(swapped_bits) if not b}
        new_cut = calc_cut_size(G, part0, part1)
        if new_cut > best_cut:
            best_cut, best_bits = new_cut, swapped_bits

    return best_bits, best_cut


def compute_nu(graph: Graph) -> float:
    """Cota de Edwards-Erdos (grafo no ponderado) o Poljak-Turzik (ponderado)."""
    ordered_nodes = sorted(graph.nodes())
    adj_matrix = nx.to_numpy_array(graph, nodelist=ordered_nodes)
    if np.any((adj_matrix != 0) & (adj_matrix != 1)):
        min_span_tree = nx.minimum_spanning_tree(graph)
        ordered_tree_nodes = sorted(min_span_tree.nodes())
        min_span_tree_matrix = nx.to_numpy_array(min_span_tree, nodelist=ordered_tree_nodes)
        tree_total_weight = np.sum(min_span_tree_matrix) / 2
        graph_weight = np.sum(adj_matrix) / 2
        nu = graph_weight / 2 + tree_total_weight / 4
    else:
        n_edges = len(graph.edges())
        n_nodes = len(graph.nodes())
        nu = n_edges / 2 + (n_nodes - 1) / 4
    return nu


# ---------------------------------------------------------------------------
# Adaptado de circuit_builder.py — Ansatz HEA "Taylor efficient"
# ---------------------------------------------------------------------------

class PCECircuit:
    """
    Versión recortada de la clase Circuit del repo PCE, fijada a la
    configuración usada en el paper para esta primera implementación:
    entanglement='Taylor_efficient', rotation='Taylor_efficient',
    connectivity='brickwork_single_rotating'.
    """

    def __init__(self, size: int, p: int):
        self.size = size
        self.p = p
        self.circuit_representation = None

    def compile_circuit(self):
        c = QuantumCircuit(self.size)
        self.circuit_representation = self._build(c)

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit_representation

    def _define_connectivity(self, layer: int) -> List[Tuple[int, int]]:
        """connectivity='brickwork_single_rotating'."""
        layer = math.ceil(layer / 2)
        entang_list_unpaired = [q for q in range(layer - 1, self.size + layer - 1)]

        def refit(lst, size):
            for i in range(len(lst)):
                if lst[i] > size - 1:
                    lst[i] -= size
            if all(q < size for q in lst):
                return lst
            return refit(lst, size)

        entang_list_unpaired = refit(entang_list_unpaired, self.size)
        entang_list = []
        for q in range(1, len(entang_list_unpaired), 2):
            entang_list.append((entang_list_unpaired[q - 1], entang_list_unpaired[q]))
        return entang_list

    def _build(self, c: QuantumCircuit) -> QuantumCircuit:
        from qiskit.circuit import ParameterVector

        param = ParameterVector("p", 0)
        params_number = 0
        counter_taylor = 0

        for l in range(self.p):
            entang_list = self._define_connectivity(l)

            if l % 2 == 0:
                # Rotaciones: alterna RZ, RX, RY (Taylor_efficient)
                rot = ["RZ", "RX", "RY"][counter_taylor % 3]
                counter_taylor += 1
                for q in range(self.size):
                    if q in [x for t in entang_list for x in t] or l == 0:
                        if len(param) < params_number + 1:
                            param.resize(params_number + 1)
                        if rot == "RX":
                            c.rx(param[params_number], q)
                        elif rot == "RY":
                            c.ry(param[params_number], q)
                        else:
                            c.rz(param[params_number], q)
                        params_number += 1
            else:
                # Entanglement: alterna RZZ, RXX, RYY (Taylor_efficient)
                ent = ["RZZ", "RXX", "RYY"][counter_taylor % 3]
                counter_taylor += 1
                for q0, q1 in entang_list:
                    if q0 is None or q1 is None:
                        continue
                    if len(param) < params_number + 1:
                        param.resize(params_number + 1)
                    if ent == "RZZ":
                        c.rzz(param[params_number], q0, q1)
                    elif ent == "RXX":
                        from qiskit.circuit.library import RXXGate
                        c.append(RXXGate(param[params_number]), [q0, q1])
                    else:
                        from qiskit.circuit.library import RYYGate
                        c.append(RYYGate(param[params_number]), [q0, q1])
                    params_number += 1

        return c


# ---------------------------------------------------------------------------
# Adaptado de tensor_exp_value.py
# ---------------------------------------------------------------------------

def computational_basis_tensor(n_circuits: int, n_qubits: int) -> np.ndarray:
    n_states = 2 ** n_qubits
    states = np.arange(n_states, dtype=np.int64)[:, None]
    bits = ((states >> np.arange(n_qubits - 1, -1, -1)) & 1).astype(np.uint8)
    return np.broadcast_to(bits, (n_circuits, n_states, n_qubits))


def combination_tensor(n_circuits: int, n_qubits: int, k_degree: int) -> np.ndarray:
    n_combinations = comb(n_qubits, k_degree)
    base = np.zeros((n_qubits, n_combinations), dtype=np.int8)
    for j, combo in enumerate(combinations(range(n_qubits), k_degree)):
        base[list(combo), j] = 1
    base = base[::-1, :]
    return np.broadcast_to(base, (n_circuits, n_qubits, n_combinations))


def build_sign_tensor(n_circuits: int, n_qubits: int, k_degree: int) -> np.ndarray:
    a = computational_basis_tensor(n_circuits, n_qubits)
    b = combination_tensor(n_circuits, n_qubits, k_degree)
    c = a @ b
    d = (-1) ** c
    return d.transpose(0, 2, 1)


def build_probability_tensor(prob_list: List[np.ndarray], n_qubits: int) -> np.ndarray:
    """Versión simplificada: prob_list ya son arrays de probabilidades exactas."""
    prob_arrays = []
    for probs in prob_list:
        probs = np.asarray(probs, dtype=float)
        if not np.isclose(probs.sum(), 1.0):
            probs = probs / probs.sum()
        prob_arrays.append(probs.reshape(-1, 1))
    return np.stack(prob_arrays, axis=0)


def run_with_probabilities(d_t: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    result = d_t @ p_t
    return result.transpose(0, 2, 1)


def select_nodes_from_aux(aux: np.ndarray, m: int, n: int) -> Dict[int, float]:
    """Selecciona m valores de expectativa: n de X, n de Y, resto de Z."""
    rem = m - 2 * n
    if rem < 0:
        raise ValueError(f"n demasiado grande: m={m}, n={n}, m-2n={rem}")
    x_vals = aux[0, 0, :n]
    y_vals = aux[1, 0, :n]
    z_vals = aux[2, 0, :rem] if rem > 0 else np.array([])
    concatenated = np.concatenate([x_vals, y_vals, z_vals])
    return {i: val for i, val in enumerate(concatenated)}


# ---------------------------------------------------------------------------
# Adaptado de loss_functions.py — versión "Simulation" (statevector exacto)
# ---------------------------------------------------------------------------

def pce_loss_func(
    x: np.ndarray,
    alpha: float,
    beta: float,
    transpiled_z: QuantumCircuit,
    transpiled_x: QuantumCircuit,
    transpiled_y: QuantumCircuit,
    sim: AerSimulator,
    graph: Graph,
    list_size: int,
    num_qubits: int,
    d_t: np.ndarray,
    experiment_result: List[Dict],
) -> float:
    """
    Evalúa la función de pérdida PCE para un set de parámetros x,
    usando statevector exacto (sin shots, sin ruido).

    OPTIMIZACIÓN: transpiled_z/x/y ya vienen transpilados al backend
    (estructura del circuito + rotaciones de base + save_statevector),
    sin parámetros asignados. Aquí solo se hace assign_parameters, que
    es mucho más barato que volver a transpilar en cada evaluación.
    """
    param_dict_z = {p: v for p, v in zip(transpiled_z.parameters, x)}
    param_dict_x = {p: v for p, v in zip(transpiled_x.parameters, x)}
    param_dict_y = {p: v for p, v in zip(transpiled_y.parameters, x)}

    bound_z = transpiled_z.assign_parameters(param_dict_z)
    bound_x = transpiled_x.assign_parameters(param_dict_x)
    bound_y = transpiled_y.assign_parameters(param_dict_y)

    state_z = sim.run(bound_z).result().get_statevector(bound_z)
    probs_z = np.abs(np.asarray(state_z)) ** 2

    state_x = sim.run(bound_x).result().get_statevector(bound_x)
    probs_x = np.abs(np.asarray(state_x)) ** 2

    state_y = sim.run(bound_y).result().get_statevector(bound_y)
    probs_y = np.abs(np.asarray(state_y)) ** 2

    p_t = build_probability_tensor([probs_x, probs_y, probs_z], num_qubits)
    aux = run_with_probabilities(d_t, p_t)

    m = graph.number_of_nodes()
    node_exp_map = select_nodes_from_aux(aux, m, list_size)

    loss = sum(
        np.tanh(alpha * node_exp_map[edge0 - 1]) * np.tanh(alpha * node_exp_map[edge1 - 1])
        for edge0, edge1 in graph.edges()
    )

    regulation_term = np.mean(
        [np.tanh(alpha * node_exp_map[i]) ** 2 for i in range(num_qubits)]
    ) ** 2
    nu = compute_nu(graph)
    loss += beta * nu * regulation_term

    experiment_result.append({"loss": loss, "exp_map": node_exp_map})
    return loss


# ---------------------------------------------------------------------------
# Clase principal: QScoreBenchmarkPCE
# ---------------------------------------------------------------------------

class QScoreBenchmarkPCE:
    """
    Versión PCE del benchmark Q-Score. Reutiliza la generación de grafos
    Erdős-Rényi de qscore.py; sustituye QAOA por el pipeline PCE.

    PRIMERA VERSIÓN: simulación exacta con AerSimulator(method='statevector'),
    sin shots, sin ruido — solo para validar que el pipeline corre bien.
    """

    def __init__(self, configuration: QScoreConfiguration):
        self.configuration = configuration
        self.num_instances = configuration.num_instances
        self.min_num_nodes = configuration.min_num_nodes
        self.max_num_nodes = configuration.max_num_nodes
        self.node_step = configuration.node_step
        self.seed = configuration.seed
        self.k = configuration.k
        self.pce_maxiter = configuration.pce_maxiter
        self.pce_optimizer = configuration.pce_optimizer
        self.pce_alpha_factor = configuration.pce_alpha_factor
        self.pce_beta = configuration.pce_beta
        self.pce_de_popsize = configuration.pce_de_popsize
        self.pce_de_strategy = configuration.pce_de_strategy
        self.pce_de_mutation = configuration.pce_de_mutation
        self.pce_de_recombination = configuration.pce_de_recombination
        self.pce_de_tol = configuration.pce_de_tol
        self.pce_de_polish = configuration.pce_de_polish
        self.pce_de_init = configuration.pce_de_init

        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

    @staticmethod
    def _relabel_from_one(graph: Graph) -> Graph:
        """nx.erdos_renyi_graph genera nodos 0..n-1; el código PCE espera 1..n."""
        mapping = {node: node + 1 for node in graph.nodes()}
        return nx.relabel_nodes(graph, mapping)

    def run_pce_on_graph(self, graph: Graph, num_ver: int) -> Tuple[int, List[int]]:
        """
        Ejecuta el pipeline PCE completo sobre un grafo y devuelve
        (cut_size_refinado, bitstring_refinado).
        """
        qubits = num_qubits(num_ver, self.k)
        layers = num_ver ** (1 - (1 / self.k))
        num_layers = math.ceil(layers)
        list_size = num_ver // 3

        alpha = self.pce_alpha_factor * qubits
        beta = self.pce_beta

        # --- Construir el ansatz (estructura, sin parámetros asignados) ---
        builder = PCECircuit(size=qubits, p=num_layers)
        builder.compile_circuit()
        base_circuit = builder.get_circuit()

        sim = AerSimulator(method="statevector")

        # --- Construir y transpilar UNA SOLA VEZ los tres circuitos de medida ---
        # (estructura + rotaciones de base + save_statevector), sin bind de
        # parámetros todavía. Esto evita re-transpilar en cada evaluación de
        # la función de pérdida, que era el cuello de botella principal.
        qc_z = base_circuit.copy()
        qc_z.save_statevector()

        qc_x = base_circuit.copy()
        for q in range(qubits):
            qc_x.h(q)
        qc_x.save_statevector()

        qc_y = base_circuit.copy()
        for q in range(qubits):
            qc_y.sdg(q)
            qc_y.h(q)
        qc_y.save_statevector()

        transpiled_z = transpile(qc_z, sim)
        transpiled_x = transpile(qc_x, sim)
        transpiled_y = transpile(qc_y, sim)

        # --- Tensor de signos precomputado ---
        d_t = build_sign_tensor(n_circuits=3, n_qubits=qubits, k_degree=self.k)

        # --- Optimización clásica ---
        experiment_result: List[Dict] = []
        rng = np.random.default_rng(33)
        n_params = len(base_circuit.parameters)
        initial_params = rng.random(n_params) * 2 * np.pi

        def loss(x):
            return pce_loss_func(
                x, alpha, beta, transpiled_z, transpiled_x, transpiled_y, sim,
                graph, list_size, qubits, d_t, experiment_result,
            )

        optimizer_lower = self.pce_optimizer.lower()

        if optimizer_lower == "differentialevolution":
            # Optimizador global por población — más robusto frente a mínimos
            # locales y plateaus de la función tanh(alpha * <Pi>), a costa de
            # bastantes más evaluaciones de la función de coste.
            result = differential_evolution(
                loss,
                bounds=[(0, 2 * np.pi)] * n_params,
                seed=33,
                strategy=self.pce_de_strategy,
                maxiter=self.pce_maxiter,
                popsize=self.pce_de_popsize,
                tol=self.pce_de_tol,
                mutation=self.pce_de_mutation,
                recombination=self.pce_de_recombination,
                polish=self.pce_de_polish,
                init=self.pce_de_init,
                disp=False,
            )
        else:
            # Optimizadores locales de scipy.optimize.minimize
            # (COBYLA, Powell, BFGS, L-BFGS-B, SLSQP, etc.)
            result = minimize(
                loss, initial_params,
                method=self.pce_optimizer,
                options={"maxiter": self.pce_maxiter},
            )

        # --- Extraer la mejor solución y refinarla localmente ---
        min_loss = min(e["loss"] for e in experiment_result)
        best = next(e for e in reversed(experiment_result) if e["loss"] == min_loss)
        par0, par1, initial_cut, bitstring_init = get_partition_from_expmap(best["exp_map"], graph)
        best_bits, best_cut = local_refinement_from_partition(graph, par0, par1)

        return int(best_cut), best_bits

    def execute(self) -> BenchmarkRunResult:
        """Genera los grafos y ejecuta PCE para cada tamaño/instancia."""
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")
        dataset = xr.Dataset()

        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend_name"] = "AerSimulator-statevector-PCE"
        for key, value in self.configuration:
            dataset.attrs[key] = value

        node_numbers = list(range(self.min_num_nodes, self.max_num_nodes + 1, self.node_step))
        dataset.attrs["node_numbers"] = node_numbers
        dataset.attrs["max_num_nodes"] = node_numbers[-1]

        for num_nodes in node_numbers:
            qcvv_logger.info(f"[PCE] Ejecutando {self.num_instances} grafos con {num_nodes} nodos.")

            seed = self.seed
            graph_list = []
            cut_sizes_list = []

            for instance in range(self.num_instances):
                # Generar grafo 0..n-1 y relabel a 1..n (formato esperado por PCE)
                graph_zero = nx.erdos_renyi_graph(num_nodes, 0.5, seed=seed)
                seed += 1
                graph = self._relabel_from_one(graph_zero)
                graph_list.append(graph)

                if graph.number_of_edges() == 0:
                    cut_sizes_list.append(0)
                    continue

                cut_size, _bits = self.run_pce_on_graph(graph, num_nodes)
                cut_sizes_list.append(cut_size)

            dataset.attrs[num_nodes] = {
                "graph": graph_list,
                "cut_sizes": cut_sizes_list,
            }

        return BenchmarkRunResult(dataset=dataset)


def qscore_pce_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Misma métrica beta que en qscore.py, aplicada a los cortes obtenidos con PCE."""
    import matplotlib.pyplot as plt

    plots = {}
    observations: List[BenchmarkObservation] = []
    dataset = run.dataset.copy(deep=True)

    backend_name = dataset.attrs["backend_name"]
    timestamp = dataset.attrs["execution_timestamp"]
    nodes_list = dataset.attrs["node_numbers"]
    num_instances = dataset.attrs["num_instances"]

    LAMBDA = 0.178
    qscore = 0
    beta_ratio_list = []
    beta_ratio_std_list = []

    for num_nodes in nodes_list:
        cut_sizes_list = dataset.attrs[num_nodes]["cut_sizes"]

        average_cut_size = np.mean(cut_sizes_list) - num_nodes * (num_nodes - 1) / 8
        average_best_cut_size = LAMBDA * pow(num_nodes, 3 / 2)
        approximation_ratio = float(average_cut_size / average_best_cut_size)

        approximation_ratio_array = (
            np.array(cut_sizes_list) - num_nodes * (num_nodes - 1) / 8
        ) / (LAMBDA * num_nodes ** (3 / 2))

        beta_ratio_list.append(np.mean(approximation_ratio_array))
        std_of_approximation_ratio = np.std(approximation_ratio_array) / np.sqrt(
            max(len(approximation_ratio_array) - 1, 1)
        )
        beta_ratio_std_list.append(std_of_approximation_ratio)

        success = is_successful(approximation_ratio)
        if success:
            qcvv_logger.info(
                f"[PCE] n={num_nodes} PASSED | beta={approximation_ratio:.4f} ± {std_of_approximation_ratio:.4f}"
            )
            qscore = num_nodes
        else:
            qcvv_logger.info(
                f"[PCE] n={num_nodes} FAILED | beta={approximation_ratio:.4f} ± {std_of_approximation_ratio:.4f} < 0.2"
            )

        observations.extend([
            BenchmarkObservation(name="mean_approximation_ratio", value=approximation_ratio,
                                 uncertainty=std_of_approximation_ratio),
            BenchmarkObservation(name="is_successful", value=str(success)),
            BenchmarkObservation(name="Qscore_result", value=qscore if success else 1),
        ])

    fig = plt.figure()
    ax = plt.axes()
    plt.axhline(0.2, color="red", linestyle="dashed", label="Threshold")
    plt.errorbar(nodes_list, beta_ratio_list, yerr=beta_ratio_std_list, fmt="-o",
                 capsize=10, markersize=8, color="#2E8B57", label="PCE approximation ratio")
    ax.set_ylabel(r"Q-score ratio $\beta(n)$")
    ax.set_xlabel("Number of nodes $(n)$")
    plt.xticks(range(min(nodes_list), max(nodes_list) + 1))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title(f"Q-score (PCE), {num_instances} instances\n{backend_name} / {timestamp}", fontsize=9)
    fig_name = f"PCE_{max(nodes_list)}_nodes_{num_instances}_instances_.png"
    plt.gcf().set_dpi(250)
    plt.close()
    plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)