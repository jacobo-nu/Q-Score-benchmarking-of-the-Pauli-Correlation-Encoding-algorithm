"""
qscore_PCE.py
=============
Version del benchmark Q-Score que sustituye QAOA por PCE (Pauli Correlation
Encoding) como algoritmo cuantico para resolver Max-Cut.

CAMBIOS 10/07/2026 (sesion de depuracion paralelismo FT3):
- get_max_parallel_threads() ya NO deriva de --cpus-per-task/num_workers;
  es una constante fija (ver docstring de la funcion para el motivo: bug
  de no-reproducibilidad del beta encontrado y confirmado experimentalmente).
- _compute_one_instance instrumentado con timestamps [TIMING] por instancia
  (pid, tiempos de grafo/init/computo) para diagnostico de rendimiento.
- _run_instances_parallel ya no reparte --cpus-per-task entre workers.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from itertools import combinations
from math import comb
from time import strftime, perf_counter
from typing import Dict, List, Tuple

import networkx as nx
from networkx import Graph
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize, differential_evolution
import xarray as xr

qcvv_logger = logging.getLogger("qscore_pce")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_AER_METADATA_LOGGED = False
_AER_ACTUAL_DEVICE = None


@dataclass
class QScoreConfiguration:
    num_instances: int = 10
    min_num_nodes: int = 2
    max_num_nodes: int = 5
    node_step: int = 1
    seed: int = 42

    k: int = 2
    pce_maxiter: int = 10
    pce_optimizer: str = "COBYLA"
    pce_alpha_factor: float = 1.5
    pce_beta: float = 0.5

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


def get_sim_device() -> str:
    return os.environ.get("PCE_DEVICE", "CPU")


def get_sim_method() -> str:
    return os.environ.get("PCE_SIM_METHOD", "statevector")


def get_shots() -> int:
    return int(os.environ.get("PCE_SHOTS", 0))


def get_backend_name() -> str:
    return os.environ.get("PCE_BACKEND", "AER").upper()


def get_qmio_backend():
    backend_name = get_backend_name()
    from qmiotools.integrations.qiskitqmio import QmioBackend, FakeQmio

    if backend_name == "QMIO_REAL":
        return QmioBackend()
    elif backend_name == "QMIO_FAKE":
        return FakeQmio()
    else:
        raise ValueError(
            f"get_qmio_backend() llamado con PCE_BACKEND={backend_name!r}; "
            "solo valido para QMIO_REAL o QMIO_FAKE."
        )


def get_max_parallel_threads() -> int:
    """
    Nº de hilos OpenMP que AerSimulator puede usar internamente
    (max_parallel_threads).

    FIJO A UN VALOR CONSTANTE (por defecto 1), NO derivado de
    --cpus-per-task ni de num_workers.

    Motivo (bug real encontrado 10/07/2026): la suma en coma flotante que
    hace AerSimulator al repartir el algebra lineal del statevector entre
    varios hilos OpenMP no es asociativa - sumar en distinto orden (que es
    justo lo que cambia al variar el nº de hilos) produce resultados que
    difieren en el ultimo bit de precision (~1e-15). Esa diferencia
    minuscula se retroalimenta dentro de differential_evolution (optimizador
    global, no convexo, muy sensible a la trayectoria de busqueda), y acaba
    produciendo un beta final distinto y REPRODUCIBLE dentro de cada config,
    pero NO COMPARABLE entre configs con distinto nº de hilos.

    Para que los resultados del benchmark sean comparables entre si
    (distintos NUM_WORKERS, distintos clusters, distintas ejecuciones),
    este valor se fija a una constante independiente de la infraestructura
    en vez de derivarse de --cpus-per-task/num_workers. Configurable via
    PCE_MAX_PARALLEL_THREADS SOLO para pruebas de rendimiento puntuales -
    NO cambiar este valor entre ejecuciones cuyo beta se vaya a comparar.

    NOTA: se confirmo experimentalmente (10/07/2026, tests --exclusive en
    ilk-244 e ilk-194) que la variabilidad de beta observada en FT3 no
    dependia en realidad de este valor ni del nodo fisico, sino de
    condiciones de contencion/turbo del nodo en el momento de ejecucion.
    Aun asi, este valor se mantiene fijo por buena practica de
    reproducibilidad y para eliminar una fuente adicional de varianza.
    """
    return int(os.environ.get("PCE_MAX_PARALLEL_THREADS", 1))


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}:{hours:02d}:{minutes:02d}:{secs:02d}"


def is_successful(approximation_ratio: float) -> bool:
    return bool(approximation_ratio > 0.2)


def cut_cost_function(x: str, graph: Graph) -> int:
    obj = 0
    for i, j in graph.edges():
        if x[i] != x[j]:
            obj += 1
    return -1 * obj


def num_qubits(num_variables: int, order_compression: int) -> int:
    list_size = num_variables // 3
    rem = num_variables - 2 * list_size

    qubits = max(order_compression, 2)
    while comb(qubits, order_compression) < rem:
        qubits += 1

    return qubits


def calc_cut_size(graph: Graph, partition0, partition1) -> float:
    cut_size = 0.0
    for edge0, edge1, data in graph.edges(data=True):
        if (edge0 in partition0 and edge1 in partition1) or \
           (edge0 in partition1 and edge1 in partition0):
            cut_size += data.get("weight", 1.0)
    return cut_size


def get_partition_from_expmap(exp_map: Dict[int, float], G: Graph):
    par0 = {i + 1 for i, val in exp_map.items() if val >= 0}
    par1 = {i + 1 for i, val in exp_map.items() if val < 0}
    cut_size = calc_cut_size(G, par0, par1)
    bitstring_init = [1 if val >= 0 else 0 for val in exp_map.values()]
    return par0, par1, cut_size, bitstring_init


def local_refinement_from_partition(G: Graph, par0, par1):
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


class PCECircuit:
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
    rem = m - 2 * n
    if rem < 0:
        raise ValueError(f"n demasiado grande: m={m}, n={n}, m-2n={rem}")
    x_vals = aux[0, 0, :n]
    y_vals = aux[1, 0, :n]
    z_vals = aux[2, 0, :rem] if rem > 0 else np.array([])
    concatenated = np.concatenate([x_vals, y_vals, z_vals])
    return {i: val for i, val in enumerate(concatenated)}


def counts_to_probs(counts: Dict[str, int], num_qubits: int, shots: int) -> np.ndarray:
    probs = np.zeros(2 ** num_qubits)
    for bitstring, count in counts.items():
        probs[int(bitstring, 2)] = count / shots
    return probs


def _circuit_probs(sim: AerSimulator, bound_circuit: QuantumCircuit, shots: int, num_qubits: int):
    if shots > 0:
        result = sim.run(bound_circuit, shots=shots).result()
        probs = counts_to_probs(result.get_counts(bound_circuit), num_qubits, shots)
    else:
        result = sim.run(bound_circuit).result()
        probs = np.abs(np.asarray(result.get_statevector(bound_circuit))) ** 2
    return result, probs


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
    shots: int,
) -> float:
    param_dict_z = {p: v for p, v in zip(transpiled_z.parameters, x)}
    param_dict_x = {p: v for p, v in zip(transpiled_x.parameters, x)}
    param_dict_y = {p: v for p, v in zip(transpiled_y.parameters, x)}

    bound_z = transpiled_z.assign_parameters(param_dict_z)
    bound_x = transpiled_x.assign_parameters(param_dict_x)
    bound_y = transpiled_y.assign_parameters(param_dict_y)

    result_z, probs_z = _circuit_probs(sim, bound_z, shots, num_qubits)

    global _AER_METADATA_LOGGED, _AER_ACTUAL_DEVICE
    if not _AER_METADATA_LOGGED:
        if get_backend_name() == "AER":
            meta = result_z.results[0].metadata
            _AER_ACTUAL_DEVICE = meta.get("device")
            qcvv_logger.info(
                f"[PCE] Aer metadata (1a ejecucion de este proceso): "
                f"device={meta.get('device')} method={meta.get('method')} "
                f"shots={shots if shots > 0 else 'exact'} "
                f"fusion_enabled={meta.get('fusion', {}).get('enabled') if isinstance(meta.get('fusion'), dict) else None}"
            )
        else:
            qcvv_logger.info(
                f"[PCE] Backend QMIO (1a ejecucion de este proceso): "
                f"backend={get_backend_name()} shots={shots if shots > 0 else 'exact'}"
            )
        _AER_METADATA_LOGGED = True

    _, probs_x = _circuit_probs(sim, bound_x, shots, num_qubits)
    _, probs_y = _circuit_probs(sim, bound_y, shots, num_qubits)

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


def rotosolve(loss_fn, initial_params: np.ndarray, maxiter: int, tol: float = 1e-6) -> np.ndarray:
    theta = np.array(initial_params, dtype=float, copy=True)
    n_params = len(theta)
    prev_loss = loss_fn(theta)

    for _sweep in range(maxiter):
        for i in range(n_params):
            theta_i = theta[i]

            loss_0 = loss_fn(theta)

            theta[i] = theta_i + np.pi / 2
            loss_plus = loss_fn(theta)

            theta[i] = theta_i - np.pi / 2
            loss_minus = loss_fn(theta)

            theta[i] = theta_i - np.pi / 2 - np.arctan2(
                2 * loss_0 - loss_plus - loss_minus,
                loss_plus - loss_minus,
            )

        current_loss = loss_fn(theta)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return theta


def compute_beta(cut_sizes_list: List[float], num_nodes: int) -> Tuple[float, float]:
    LAMBDA = 0.178
    average_cut_size = np.mean(cut_sizes_list) - num_nodes * (num_nodes - 1) / 8
    average_best_cut_size = LAMBDA * pow(num_nodes, 3 / 2)
    approximation_ratio = float(average_cut_size / average_best_cut_size)

    approximation_ratio_array = (
        np.array(cut_sizes_list) - num_nodes * (num_nodes - 1) / 8
    ) / (LAMBDA * num_nodes ** (3 / 2))
    beta_std = float(np.std(approximation_ratio_array) / np.sqrt(
        max(len(approximation_ratio_array) - 1, 1)
    ))
    return approximation_ratio, beta_std


def _compute_one_instance(args: tuple) -> Tuple[int, int]:
    """
    NOTA 10/07/2026: ya NO recibe threads_per_worker (se quito de la
    tupla de argumentos) - max_parallel_threads es ahora una constante
    fija (ver get_max_parallel_threads()), no depende de num_workers.
    """
    t_start = perf_counter()
    pid = os.getpid()

    (num_nodes, instance_index, base_seed, k, pce_maxiter, pce_optimizer,
     pce_alpha_factor, pce_beta, pce_de_popsize, pce_de_strategy,
     pce_de_mutation, pce_de_recombination, pce_de_tol, pce_de_polish,
     pce_de_init) = args

    # Salvaguarda para librerias que puedan leer SLURM_CPUS_PER_TASK para
    # autoconfigurar su propio paralelismo interno. Se fija al MISMO valor
    # constante que get_max_parallel_threads() (no derivado de
    # num_workers/threads_per_worker).
    os.environ["SLURM_CPUS_PER_TASK"] = str(get_max_parallel_threads())

    seed = base_seed + instance_index
    graph_zero = nx.erdos_renyi_graph(num_nodes, 0.5, seed=seed)
    graph = QScoreBenchmarkPCE._relabel_from_one(graph_zero)

    if graph.number_of_edges() == 0:
        return instance_index, 0

    t_graph = perf_counter()

    config = QScoreConfiguration(
        num_instances=1, min_num_nodes=num_nodes, max_num_nodes=num_nodes, node_step=1,
        seed=base_seed, k=k, pce_maxiter=pce_maxiter, pce_optimizer=pce_optimizer,
        pce_alpha_factor=pce_alpha_factor, pce_beta=pce_beta,
        pce_de_popsize=pce_de_popsize, pce_de_strategy=pce_de_strategy,
        pce_de_mutation=pce_de_mutation, pce_de_recombination=pce_de_recombination,
        pce_de_tol=pce_de_tol, pce_de_polish=pce_de_polish, pce_de_init=pce_de_init,
    )
    bench = QScoreBenchmarkPCE(configuration=config)

    t_init = perf_counter()

    cut_size, _bits = bench.run_pce_on_graph(graph, num_nodes)

    t_end = perf_counter()

    print(
        f"[TIMING] pid={pid} idx={instance_index} "
        f"t0={t_start:.3f} "
        f"grafo={t_graph - t_start:.3f}s "
        f"init_bench={t_init - t_graph:.3f}s "
        f"run_pce={t_end - t_init:.3f}s "
        f"total={t_end - t_start:.3f}s",
        flush=True,
    )
    return instance_index, cut_size


class QScoreBenchmarkPCE:
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
        mapping = {node: node + 1 for node in graph.nodes()}
        return nx.relabel_nodes(graph, mapping)

    def run_pce_on_graph(self, graph: Graph, num_ver: int) -> Tuple[int, List[int]]:
        qubits = num_qubits(num_ver, self.k)
        layers = num_ver ** (1 - (1 / self.k))
        num_layers = math.ceil(layers)
        list_size = num_ver // 3

        alpha = self.pce_alpha_factor * qubits
        beta = self.pce_beta

        builder = PCECircuit(size=qubits, p=num_layers)
        builder.compile_circuit()
        base_circuit = builder.get_circuit()

        backend_name = get_backend_name()
        shots = get_shots()

        if backend_name in ("QMIO_REAL", "QMIO_FAKE"):
            if shots <= 0:
                raise ValueError(
                    f"PCE_BACKEND={backend_name} requiere PCE_SHOTS>0 "
                    "(ni el hardware real ni FakeQmio pueden leer un "
                    "statevector exacto, solo shots)."
                )
            sim = get_qmio_backend()
        else:
            sim = AerSimulator(
                method=get_sim_method(),
                device=get_sim_device(),
                max_parallel_threads=get_max_parallel_threads(),
            )

        qc_z = base_circuit.copy()
        qc_x = base_circuit.copy()
        qc_y = base_circuit.copy()

        for q in range(qubits):
            qc_x.h(q)
        for q in range(qubits):
            qc_y.sdg(q)
            qc_y.h(q)

        if shots > 0:
            qc_z.measure_all()
            qc_x.measure_all()
            qc_y.measure_all()
        else:
            qc_z.save_statevector()
            qc_x.save_statevector()
            qc_y.save_statevector()

        transpiled_z = transpile(qc_z, sim)
        transpiled_x = transpile(qc_x, sim)
        transpiled_y = transpile(qc_y, sim)

        d_t = build_sign_tensor(n_circuits=3, n_qubits=qubits, k_degree=self.k)

        experiment_result: List[Dict] = []
        rng = np.random.default_rng(33)
        n_params = len(base_circuit.parameters)
        initial_params = rng.random(n_params) * 2 * np.pi

        def loss(x):
            return pce_loss_func(
                x, alpha, beta, transpiled_z, transpiled_x, transpiled_y, sim,
                graph, list_size, qubits, d_t, experiment_result, shots,
            )

        optimizer_lower = self.pce_optimizer.lower()

        if optimizer_lower == "differentialevolution":
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
        elif optimizer_lower == "rotosolve":
            rotosolve(loss, initial_params, maxiter=self.pce_maxiter)
        else:
            result = minimize(
                loss, initial_params,
                method=self.pce_optimizer,
                options={"maxiter": self.pce_maxiter},
            )

        min_loss = min(e["loss"] for e in experiment_result)
        best = next(e for e in reversed(experiment_result) if e["loss"] == min_loss)
        par0, par1, initial_cut, bitstring_init = get_partition_from_expmap(best["exp_map"], graph)
        best_bits, best_cut = local_refinement_from_partition(graph, par0, par1)

        return int(best_cut), best_bits

    def _seed_for_instance(self, num_nodes: int, instance_index: int) -> int:
        return self.seed + num_nodes * 100_000 + instance_index

    def _run_instances_serial(self, num_nodes: int, instance_indices: List[int]) -> List[int]:
        start_time = perf_counter()
        cut_sizes = []
        total = len(instance_indices)
        for pos, idx in enumerate(instance_indices):
            instance_start = perf_counter()

            seed = self._seed_for_instance(num_nodes, idx)
            graph_zero = nx.erdos_renyi_graph(num_nodes, 0.5, seed=seed)
            graph = self._relabel_from_one(graph_zero)

            if graph.number_of_edges() == 0:
                cut_sizes.append(0)
            else:
                cut_size, _bits = self.run_pce_on_graph(graph, num_nodes)
                cut_sizes.append(cut_size)

            completed = pos + 1
            instance_duration = perf_counter() - instance_start
            elapsed = perf_counter() - start_time
            avg_per_instance = elapsed / completed
            remaining_estimate = avg_per_instance * (total - completed)
            qcvv_logger.info(
                f"[PCE] n={num_nodes} instancia (idx={idx}) {completed}/{total} "
                f"completada en {format_duration(instance_duration)} - "
                f"transcurrido={format_duration(elapsed)} "
                f"restante_estimado={format_duration(remaining_estimate)}"
            )
        return cut_sizes

    def _run_instances_parallel(self, num_nodes: int, instance_indices: List[int], num_workers: int) -> List[int]:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        base_seed = self.seed + num_nodes * 100_000

        qcvv_logger.info(
            f"[PCE] n={num_nodes} paralelizando {len(instance_indices)} instancias entre "
            f"{num_workers} workers (max_parallel_threads={get_max_parallel_threads()} fijo, "
            f"independiente de num_workers)"
        )

        tasks = [
            (num_nodes, idx, base_seed, self.k, self.pce_maxiter, self.pce_optimizer,
             self.pce_alpha_factor, self.pce_beta, self.pce_de_popsize, self.pce_de_strategy,
             self.pce_de_mutation, self.pce_de_recombination, self.pce_de_tol, self.pce_de_polish,
             self.pce_de_init)
            for idx in instance_indices
        ]

        start_time = perf_counter()
        results_by_idx: Dict[int, int] = {}
        completed = 0
        total = len(instance_indices)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_compute_one_instance, t) for t in tasks]
            for future in as_completed(futures):
                idx, cut_size = future.result()
                results_by_idx[idx] = cut_size
                completed += 1
                elapsed = perf_counter() - start_time
                qcvv_logger.info(
                    f"[PCE] n={num_nodes} instancia (idx={idx}) {completed}/{total} "
                    f"completada (paralelo) - transcurrido={format_duration(elapsed)}"
                )

        return [results_by_idx[idx] for idx in instance_indices]

    def execute_single_n(self, num_nodes: int) -> Dict:
        qcvv_logger.info(f"[PCE] Ejecutando {self.num_instances} grafos con {num_nodes} nodos.")

        qubits = num_qubits(num_nodes, self.k)
        qcvv_logger.info(f"[PCE] n={num_nodes} usa {qubits} qubits (k={self.k}).")

        start_time = perf_counter()
        cut_sizes_list = self._run_instances_serial(num_nodes, list(range(self.num_instances)))
        duration_seconds = perf_counter() - start_time

        approximation_ratio, beta_std = compute_beta(cut_sizes_list, num_nodes)

        success = is_successful(approximation_ratio)
        status = "PASSED" if success else "FAILED"
        shots = get_shots()
        backend_name = get_backend_name()
        qcvv_logger.info(
            f"[PCE] n={num_nodes} beta={approximation_ratio:.4f} +/- {beta_std:.4f} ({status}) "
            f"qubits={qubits} duration={format_duration(duration_seconds)} "
            f"max_parallel_threads={get_max_parallel_threads()} "
            f"shots={shots if shots > 0 else 'exact'} backend={backend_name}"
        )

        return {
            "num_nodes": num_nodes,
            "num_qubits": qubits,
            "num_instances": self.num_instances,
            "cut_sizes": cut_sizes_list,
            "beta": approximation_ratio,
            "beta_std": beta_std,
            "success": success,
            "timestamp": strftime("%Y%m%d-%H%M%S"),
            "duration_seconds": duration_seconds,
            "k": self.k,
            "seed": self.seed,
            "pce_optimizer": self.pce_optimizer,
            "pce_maxiter": self.pce_maxiter,
            "simulator": f"AerSimulator-{get_sim_method()}",
            "device": get_sim_device(),
            "device_actual": _AER_ACTUAL_DEVICE,
            "max_parallel_threads": get_max_parallel_threads(),
            "shots": get_shots(),
            "backend": backend_name,
        }

    def execute_single_n_chunk(self, num_nodes: int, chunk_index: int, num_chunks: int,
                                num_workers: int = 1) -> Dict:
        if get_backend_name() == "QMIO_REAL" and (num_chunks > 1 or num_workers > 1):
            raise ValueError(
                "PCE_BACKEND=QMIO_REAL no admite paralelismo (ni num_chunks>1 ni "
                "num_workers>1) - un solo chip fisico, acceso sincrono. "
                "Usa num_chunks=1, num_workers=1 con este backend."
            )

        qubits = num_qubits(num_nodes, self.k)
        instance_indices = [i for i in range(self.num_instances) if i % num_chunks == chunk_index]

        qcvv_logger.info(
            f"[PCE] n={num_nodes} chunk {chunk_index + 1}/{num_chunks}: "
            f"{len(instance_indices)} instancias {instance_indices}, "
            f"num_workers={num_workers}, qubits={qubits}"
        )

        start_time = perf_counter()
        if num_workers > 1:
            cut_sizes = self._run_instances_parallel(num_nodes, instance_indices, num_workers)
        else:
            cut_sizes = self._run_instances_serial(num_nodes, instance_indices)
        duration_seconds = perf_counter() - start_time

        chunk_beta, chunk_beta_std = compute_beta(cut_sizes, num_nodes)

        shots = get_shots()
        backend_name = get_backend_name()
        qcvv_logger.info(
            f"[PCE] n={num_nodes} chunk {chunk_index + 1}/{num_chunks} completado - "
            f"beta_parcial={chunk_beta:.4f} duration={format_duration(duration_seconds)} "
            f"shots={shots if shots > 0 else 'exact'} backend={backend_name}"
        )

        return {
            "num_nodes": num_nodes,
            "num_qubits": qubits,
            "num_instances": self.num_instances,
            "chunk_index": chunk_index,
            "num_chunks": num_chunks,
            "num_workers": num_workers,
            "instance_indices": instance_indices,
            "cut_sizes": cut_sizes,
            "beta_chunk": chunk_beta,
            "beta_std_chunk": chunk_beta_std,
            "timestamp": strftime("%Y%m%d-%H%M%S"),
            "duration_seconds": duration_seconds,
            "k": self.k,
            "seed": self.seed,
            "pce_optimizer": self.pce_optimizer,
            "pce_maxiter": self.pce_maxiter,
            "simulator": f"AerSimulator-{get_sim_method()}",
            "device": get_sim_device(),
            "device_actual": _AER_ACTUAL_DEVICE,
            "max_parallel_threads": get_max_parallel_threads(),
            "shots": get_shots(),
            "backend": backend_name,
        }

    @staticmethod
    def save_result_json(result: Dict, outdir: str) -> str:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"n_{result['num_nodes']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return path

    @staticmethod
    def save_result_hdf5(result: Dict, outdir: str, filename: str = None) -> str:
        import h5py

        os.makedirs(outdir, exist_ok=True)
        if filename is None:
            filename = f"n_{result['num_nodes']}.h5"
        path = os.path.join(outdir, filename)
        with h5py.File(path, "w") as f:
            f.create_dataset("cut_sizes", data=np.asarray(result["cut_sizes"]))
            for key, value in result.items():
                if key == "cut_sizes":
                    continue
                if value is None:
                    continue
                f.attrs[key] = value
        return path

    def execute(self) -> BenchmarkRunResult:
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")
        dataset = xr.Dataset()

        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["simulator"] = f"AerSimulator-{get_sim_method()}"
        dataset.attrs["device"] = get_sim_device()
        dataset.attrs["shots"] = get_shots()
        for key, value in self.configuration:
            dataset.attrs[key] = value

        node_numbers = list(range(self.min_num_nodes, self.max_num_nodes + 1, self.node_step))
        dataset.attrs["node_numbers"] = node_numbers
        dataset.attrs["max_num_nodes"] = node_numbers[-1]

        for num_nodes in node_numbers:
            result = self.execute_single_n(num_nodes)
            dataset.attrs[num_nodes] = {
                "cut_sizes": result["cut_sizes"],
                "num_qubits": result["num_qubits"],
                "duration_seconds": result["duration_seconds"],
            }

        return BenchmarkRunResult(dataset=dataset)


def qscore_pce_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    import matplotlib.pyplot as plt

    plots = {}
    observations: List[BenchmarkObservation] = []
    dataset = run.dataset.copy(deep=True)

    simulator = dataset.attrs.get("simulator")
    device = dataset.attrs.get("device")
    shots = dataset.attrs.get("shots", 0)
    timestamp = dataset.attrs["execution_timestamp"]
    nodes_list = dataset.attrs["node_numbers"]
    num_instances = dataset.attrs["num_instances"]
    k = dataset.attrs["k"]
    optimizer = dataset.attrs["pce_optimizer"]
    maxiter = dataset.attrs["pce_maxiter"]
    seed = dataset.attrs["seed"]

    LAMBDA = 0.178
    qscore = 0
    beta_ratio_list = []
    beta_ratio_std_list = []
    qubits_list = []

    for num_nodes in nodes_list:
        cut_sizes_list = dataset.attrs[num_nodes]["cut_sizes"]
        qubits = dataset.attrs[num_nodes].get("num_qubits")
        qubits_list.append(qubits)
        duration_seconds = dataset.attrs[num_nodes].get("duration_seconds")

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
        qubits_str = f" qubits={qubits}" if qubits is not None else ""
        duration_str = f" duration={format_duration(duration_seconds)}" if duration_seconds is not None else ""
        if success:
            qcvv_logger.info(
                f"[PCE] n={num_nodes} PASSED | beta={approximation_ratio:.4f} +/- {std_of_approximation_ratio:.4f}{qubits_str}{duration_str}"
            )
            qscore = num_nodes
        else:
            qcvv_logger.info(
                f"[PCE] n={num_nodes} FAILED | beta={approximation_ratio:.4f} +/- {std_of_approximation_ratio:.4f} < 0.2{qubits_str}{duration_str}"
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

    if all(q is not None for q in qubits_list):
        ax.annotate(
            f"{qubits_list[0]} qubits",
            xy=(nodes_list[0], 1), xycoords=("data", "axes fraction"),
            xytext=(2, -10), textcoords="offset points",
            fontsize=7, color="black", rotation=90, va="top",
        )

        first_jump_labeled = False
        for i in range(1, len(qubits_list)):
            if qubits_list[i] > qubits_list[i - 1]:
                n_jump = nodes_list[i]
                label = "Qubit count increase" if not first_jump_labeled else None
                plt.axvline(x=n_jump, color="blue", linestyle="dotted", alpha=0.7, label=label)
                ax.annotate(
                    f"{qubits_list[i]} qubits",
                    xy=(n_jump, 1), xycoords=("data", "axes fraction"),
                    xytext=(2, -10), textcoords="offset points",
                    fontsize=7, color="blue", rotation=90, va="top",
                )
                first_jump_labeled = True

    ax.set_ylabel(r"Q-score ratio $\beta(n)$")
    ax.set_xlabel("Number of nodes $(n)$")
    plt.xticks(range(min(nodes_list), max(nodes_list) + 1), rotation=90)
    plt.legend(loc="lower left")
    plt.grid(True)
    title = (
        f"Q-score (PCE), {num_instances} instances\n"
        f"k={k}, optimizer={optimizer}, maxiter={maxiter}, seed={seed}"
    )
    if shots:
        title += f"\nshots={shots}" + (f" ({device})" if device is not None else "")
    elif simulator is not None:
        title += f"\n{simulator}" + (f" ({device})" if device is not None else "")
    plt.title(title, fontsize=9)
    fig_name = (
        f"PCE_n{min(nodes_list)}-{max(nodes_list)}_k{k}_{optimizer}"
        f"_maxiter{maxiter}_seed{seed}_{timestamp}.png"
    )
    plt.gcf().set_dpi(250)
    plt.close()
    plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)