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

# ---------------------------------------------------------------------------
# Reutilizado de qscore.py
# ---------------------------------------------------------------------------

qcvv_logger = logging.getLogger("qscore_pce")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Se pone a True la primera vez que se loguean los metadatos de Aer
# (device/método realmente usado), para no repetirlo en cada evaluación
# de la función de pérdida — una vez por proceso (por tarea de SLURM) basta.
_AER_METADATA_LOGGED = False
# Guarda el device que Aer confirma haber usado de verdad (via metadata),
# para poder compararlo con lo pedido en PCE_DEVICE y detectar si hubo un
# fallback silencioso a CPU. Se propaga al JSON/HDF5 de cada n.
_AER_ACTUAL_DEVICE = None


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


def get_sim_device() -> str:
    """
    Dispositivo de simulación: CPU (QMIO) o GPU (FT3, vía cuStateVec/cuQuantum).
    Controlado por la variable de entorno PCE_DEVICE, definida en el job.sh
    del cluster correspondiente. Por defecto CPU.
    """
    return os.environ.get("PCE_DEVICE", "CPU")


def get_sim_method() -> str:
    """
    Método de simulación de AerSimulator ('statevector', 'density_matrix',
    'matrix_product_state', 'tensor_network', ...). Controlado por la
    variable de entorno PCE_SIM_METHOD para poder probar otros métodos de
    Aer sin tocar el código. Por defecto 'statevector' (exacto, sin shots).
    """
    return os.environ.get("PCE_SIM_METHOD", "statevector")


def get_shots() -> int:
    """
    Número de shots por circuito (medida con muestreo estadístico, en vez
    de lectura exacta del statevector). Controlado por la variable de
    entorno PCE_SHOTS. 0 (o no definida) = modo exacto, comportamiento
    idéntico al que había antes de implementar shots.
    """
    return int(os.environ.get("PCE_SHOTS", 0))


def get_backend_name() -> str:
    """
    Backend a usar para las tres bases de medida (X/Y/Z) del pipeline PCE:
      - "AER" (por defecto): AerSimulator normal, exactamente como hasta
        ahora — controlado por PCE_DEVICE/PCE_SIM_METHOD/PCE_SHOTS.
      - "QMIO_REAL": hardware real de QMIO, vía qmiotools.integrations
        .qiskitqmio.QmioBackend. Solo funciona lanzado en la partición
        'qpu' de QMIO, y EN SERIE — nunca en un array/paralelo, la cola
        de hardware es un recurso compartido muy limitado (ejecución
        síncrona: cada .run() bloquea hasta tener resultado).
      - "QMIO_FAKE": emulador de QMIO con ruido calibrado de la última
        calibración real (qmiotools FakeQmio). A diferencia de QMIO_REAL,
        no necesita la partición 'qpu' — internamente es un AerSimulator
        normal con un modelo de ruido ya configurado, así que corre en
        cualquier nodo de CPU (p.ej. la partición 'ilk' habitual).
    Controlado por la variable de entorno PCE_BACKEND.
    """
    return os.environ.get("PCE_BACKEND", "AER").upper()


def get_qmio_backend():
    """
    Construye el backend real (QMIO_REAL) o el emulador con ruido
    (QMIO_FAKE) de QMIO, según get_backend_name().

    Import perezoso (dentro de la función, no al principio del fichero):
    el paquete qmiotools solo está disponible en el entorno de QMIO, no en
    FT3 — así el resto del código no rompe al importarse en FT3 aunque
    nunca se llegue a llamar a esta función allí.

    NO VERIFICADO EN EJECUCIÓN REAL — basado en la documentación oficial
    de qmiotools 0.2.1 (misma versión que ya tienes cargada en QMIO).
    Antes de usar esto contra hardware real, prueba primero con un
    circuito suelto pequeño (ver ejemplo en la documentación de este
    cambio) para confirmar que la integración funciona en tu entorno.
    """
    backend_name = get_backend_name()
    from qmiotools.integrations.qiskitqmio import QmioBackend, FakeQmio

    if backend_name == "QMIO_REAL":
        return QmioBackend()
    elif backend_name == "QMIO_FAKE":
        return FakeQmio()
    else:
        raise ValueError(
            f"get_qmio_backend() llamado con PCE_BACKEND={backend_name!r}; "
            "solo válido para QMIO_REAL o QMIO_FAKE."
        )


def get_max_parallel_threads() -> int:
    """
    Nº de hilos OpenMP que AerSimulator puede usar internamente
    (max_parallel_threads) — tanto para paralelizar el álgebra lineal del
    statevector como, más adelante, para repartir shots (max_parallel_shots
    reutiliza este mismo pool de hilos).

    Se toma de SLURM_CPUS_PER_TASK, que SLURM define automáticamente con
    el valor pedido en --cpus-per-task — así no hay que hardcodear el
    número ni mantenerlo sincronizado a mano con el .sh. Si se ejecuta
    fuera de SLURM (pruebas locales), cae a 1 hilo por seguridad.
    """
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


def format_duration(seconds: float) -> str:
    """
    Convierte segundos a formato 'd:hh:mm:ss', para que las duraciones de
    simulaciones largas (varias horas) sean legibles de un vistazo en logs
    y tablas, en vez de un número de segundos difícil de interpretar.
    El valor exacto en segundos se sigue guardando aparte en el JSON
    ('duration_seconds') para cálculos/gráficas.
    """
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}:{hours:02d}:{minutes:02d}:{secs:02d}"


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

def num_qubits(num_variables: int, order_compression: int) -> int:
    """
    Número de qubits PCE necesarios para num_variables variables, orden k.

    select_nodes_from_aux reparte las m variables en tres bases:
    list_size = m // 3 valores de X, list_size de Y, y
    rem = m - 2*list_size de Z. Cada base solo dispone de
    comb(qubits, k) valores, así que hace falta garantizar
    comb(qubits, k) >= rem (el mayor de los tres bloques),
    no solo 3*comb(qubits, k) >= m como suma global.

    Válido para cualquier order_compression k >= 1. Búsqueda incremental
    pura: parte de qubits = max(k, 2) (con menos de k qubits, comb(qubits,k)
    es 0, así que no tiene sentido probar valores menores) y va subiendo de
    uno en uno hasta que comb(qubits, k) cubre rem. No usa ninguna fórmula
    cerrada — la que había antes (solve_quadratic) solo era válida para
    k=2, por eso la versión anterior de esta función solo soportaba k=2 y
    lanzaba NotImplementedError para cualquier otro valor.
    """
    list_size = num_variables // 3
    rem = num_variables - 2 * list_size

    qubits = max(order_compression, 2)
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


def counts_to_probs(counts: Dict[str, int], num_qubits: int, shots: int) -> np.ndarray:
    """
    Convierte un dict de conteos (qiskit Result.get_counts()) en un vector
    denso de probabilidades de tamaño 2**num_qubits, en el mismo formato
    que antes producía get_statevector() + |amplitud|**2 — así el resto
    del pipeline (build_probability_tensor, run_with_probabilities...) no
    necesita saber si viene de shots o de statevector exacto.

    La convención de bits de Qiskit (qubit 0 = bit menos significativo)
    coincide con el índice que usa el propio array de Statevector, así
    que int(bitstring, 2) ya da el índice correcto sin invertir la cadena.
    """
    probs = np.zeros(2 ** num_qubits)
    for bitstring, count in counts.items():
        probs[int(bitstring, 2)] = count / shots
    return probs


def _circuit_probs(sim: AerSimulator, bound_circuit: QuantumCircuit, shots: int, num_qubits: int):
    """
    Ejecuta un circuito ya con parámetros asignados y devuelve (result, probs).
    shots=0 -> modo exacto (statevector, comportamiento original sin cambios).
    shots>0 -> modo muestreo (measure_all + conteos -> probabilidades estimadas).
    """
    if shots > 0:
        result = sim.run(bound_circuit, shots=shots).result()
        probs = counts_to_probs(result.get_counts(bound_circuit), num_qubits, shots)
    else:
        result = sim.run(bound_circuit).result()
        probs = np.abs(np.asarray(result.get_statevector(bound_circuit))) ** 2
    return result, probs


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
    shots: int,
) -> float:
    """
    Evalúa la función de pérdida PCE para un set de parámetros x.
    shots=0 -> statevector exacto (comportamiento original, sin ruido).
    shots>0 -> muestreo (measure_all + conteos), con el ruido estadístico
    propio de un número finito de repeticiones, más parecido a hardware real.

    OPTIMIZACIÓN: transpiled_z/x/y ya vienen transpilados al backend
    (estructura del circuito + rotaciones de base + save_statevector o
    measure_all según el modo), sin parámetros asignados. Aquí solo se
    hace assign_parameters, que es mucho más barato que volver a
    transpilar en cada evaluación.
    """
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
            # Metadatos que Aer devuelve con CADA Result, confirmando qué
            # device/método se usó de verdad (no solo lo que se pidió al
            # crear el AerSimulator) — es la forma fiable de verificar que
            # la GPU está siendo realmente utilizada, no solo aceptada sin
            # fallo. Estructura específica de Aer, no se asume para
            # QMIO_REAL (QmioJob puede no tener este mismo formato).
            meta = result_z.results[0].metadata
            _AER_ACTUAL_DEVICE = meta.get("device")
            qcvv_logger.info(
                f"[PCE] Aer metadata (1ª ejecución de este proceso): "
                f"device={meta.get('device')} method={meta.get('method')} "
                f"shots={shots if shots > 0 else 'exact'} "
                f"fusion_enabled={meta.get('fusion', {}).get('enabled') if isinstance(meta.get('fusion'), dict) else None}"
            )
        else:
            # QMIO_REAL / QMIO_FAKE: no se asume la estructura de metadata
            # de Aer (no verificada contra QmioJob). device_actual se deja
            # en None a propósito — no hay "fallback silencioso" que
            # detectar aquí, el backend ya es explícito.
            qcvv_logger.info(
                f"[PCE] Backend QMIO (1ª ejecución de este proceso): "
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


# ---------------------------------------------------------------------------
# Optimizador Rotosolve — alternativa libre de gradiente a DIFFERENTIALEVOLUTION/COBYLA
# ---------------------------------------------------------------------------

def rotosolve(loss_fn, initial_params: np.ndarray, maxiter: int, tol: float = 1e-6) -> np.ndarray:
    """
    Optimizador Rotosolve (Ostaszewski, Grant & Benedetti, "Structure
    optimization for parameterized quantum circuits", Quantum 5, 391 (2021),
    arXiv:1905.09692 — la fórmula de actualización de un solo parámetro que
    se usa aquí, sin la parte de selección de estructura del paper, es la
    que en trabajos posteriores se cita como "Rotosolve" propiamente).

    Es libre de gradiente: no estima ninguna derivada. Aprovecha que, para
    una puerta de rotación con generador auto-inverso (Pauli: RX, RY, RZ,
    RXX, RYY, RZZ — exactamente las que usa PCECircuit), la función de
    pérdida en función de UN SOLO parámetro theta_i, dejando el resto fijos,
    es exactamente sinusoidal de una única frecuencia:

        L(theta_i) = a * sin(theta_i + b) + c

    Con solo 3 evaluaciones (en theta_i, theta_i + pi/2, theta_i - pi/2) se
    reconstruye esa sinusoide por completo y se puede saltar directamente a
    su mínimo global, sin necesidad de gradiente ni de búsqueda iterativa en
    esa dirección. Se repite parámetro a parámetro, en barridos sucesivos
    (coordinate descent), hasta agotar maxiter barridos o converger.

    IMPORTANTE — interpretación de maxiter: aquí es el número de BARRIDOS
    COMPLETOS sobre todos los parámetros, no de "iteraciones" en el sentido
    de scipy.optimize.minimize/differential_evolution. Cada barrido cuesta
    3 * n_params evaluaciones de loss_fn (frente a las 3 evaluaciones de
    circuito que ya hace cada llamada a loss_fn, por las bases X/Y/Z).

    loss_fn ya se encarga de rellenar experiment_result como efecto
    secundario (ver pce_loss_func) en cada llamada — por eso esta función
    no necesita devolver ni gestionar el mejor punto visto: el resto de
    run_pce_on_graph lo extrae de experiment_result exactamente igual que
    para differential_evolution o minimize.

    ADVERTENCIA sobre PCE_SHOTS>0: la fórmula asume que las 3 evaluaciones
    son exactas. Con shots, cada evaluación tiene ruido estadístico
    binomial, así que la sinusoide reconstruida (y el salto al "óptimo")
    puede quedar sesgada por ese ruido — a diferencia de differential_evolution,
    Rotosolve no tiene ningún mecanismo interno de suavizado frente a esto
    en esta primera implementación.
    """
    theta = np.array(initial_params, dtype=float, copy=True)
    n_params = len(theta)
    prev_loss = loss_fn(theta)

    for _sweep in range(maxiter):
        for i in range(n_params):
            theta_i = theta[i]

            loss_0 = loss_fn(theta)  # theta[i] == theta_i en este punto

            theta[i] = theta_i + np.pi / 2
            loss_plus = loss_fn(theta)

            theta[i] = theta_i - np.pi / 2
            loss_minus = loss_fn(theta)

            # Fórmula cerrada de actualización (Ostaszewski et al. 2021, Eq. 5-6):
            # salta directamente al mínimo de la sinusoide en esta dirección.
            theta[i] = theta_i - np.pi / 2 - np.arctan2(
                2 * loss_0 - loss_plus - loss_minus,
                loss_plus - loss_minus,
            )

        current_loss = loss_fn(theta)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return theta


# ---------------------------------------------------------------------------
# Clase principal: QScoreBenchmarkPCE
# ---------------------------------------------------------------------------

def compute_beta(cut_sizes_list: List[float], num_nodes: int) -> Tuple[float, float]:
    """
    Calcula (approximation_ratio, beta_std) a partir de una lista de
    cut_sizes para un num_nodes dado. Extraído a función independiente
    para poder reutilizarlo tanto en execute_single_n (cálculo normal,
    serie) como en execute_single_n_chunk (informativo, por chunk) y en
    el fusionador de chunks (cálculo final, autoritativo, sobre TODAS
    las instancias ya combinadas) — misma fórmula en los tres sitios,
    sin duplicar código.
    """
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
    Función de NIVEL DE MÓDULO (no un método) — necesario para que
    ProcessPoolExecutor pueda enviarla a otros procesos sin depender de
    picklear una instancia completa de QScoreBenchmarkPCE. Calcula UNA
    única instancia (un grafo) de forma totalmente autocontenida.

    Recibe una tupla de valores simples (todos picklables sin problema) y
    devuelve (instance_index, cut_size), para que quien reciba el
    resultado sepa a qué instancia corresponde sin depender del orden de
    llegada (con multiproceso, las instancias NO terminan necesariamente
    en el mismo orden en que se lanzaron).

    Reconstruye una QScoreBenchmarkPCE "de usar y tirar" solo para poder
    llamar a run_pce_on_graph — así toda la lógica del circuito/optimizador
    se reutiliza tal cual, sin duplicar nada aquí.
    """
    (num_nodes, instance_index, base_seed, k, pce_maxiter, pce_optimizer,
     pce_alpha_factor, pce_beta, pce_de_popsize, pce_de_strategy,
     pce_de_mutation, pce_de_recombination, pce_de_tol, pce_de_polish,
     pce_de_init, threads_per_worker) = args

    # Cada proceso worker debe ver SOLO los hilos que le tocan a él, no
    # los --cpus-per-task completos de la tarea SLURM (que ya se están
    # repartiendo entre num_workers procesos simultáneos) — si no, todos
    # los workers pedirían el mismo nº de hilos completo y se
    # sobre-suscribirían los cores compitiendo entre sí.
    os.environ["SLURM_CPUS_PER_TASK"] = str(threads_per_worker)

    seed = base_seed + instance_index
    graph_zero = nx.erdos_renyi_graph(num_nodes, 0.5, seed=seed)
    graph = QScoreBenchmarkPCE._relabel_from_one(graph_zero)

    if graph.number_of_edges() == 0:
        return instance_index, 0

    config = QScoreConfiguration(
        num_instances=1, min_num_nodes=num_nodes, max_num_nodes=num_nodes, node_step=1,
        seed=base_seed, k=k, pce_maxiter=pce_maxiter, pce_optimizer=pce_optimizer,
        pce_alpha_factor=pce_alpha_factor, pce_beta=pce_beta,
        pce_de_popsize=pce_de_popsize, pce_de_strategy=pce_de_strategy,
        pce_de_mutation=pce_de_mutation, pce_de_recombination=pce_de_recombination,
        pce_de_tol=pce_de_tol, pce_de_polish=pce_de_polish, pce_de_init=pce_de_init,
    )
    bench = QScoreBenchmarkPCE(configuration=config)
    cut_size, _bits = bench.run_pce_on_graph(graph, num_nodes)
    return instance_index, cut_size


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

        # Backend a usar: AerSimulator normal (por defecto), o hardware
        # real / emulador con ruido de QMIO (ver get_backend_name()).
        backend_name = get_backend_name()
        shots = get_shots()

        if backend_name in ("QMIO_REAL", "QMIO_FAKE"):
            # Ni el hardware real ni el emulador con ruido pueden devolver
            # un statevector exacto — solo shots. PCE_SHOTS=0 (modo exacto)
            # no tiene sentido aquí, así que se exige explícitamente en vez
            # de caer en un valor por defecto silencioso que podría
            # confundirse con "modo exacto real".
            if shots <= 0:
                raise ValueError(
                    f"PCE_BACKEND={backend_name} requiere PCE_SHOTS>0 "
                    "(ni el hardware real ni FakeQmio pueden leer un "
                    "statevector exacto, solo shots)."
                )
            sim = get_qmio_backend()
        else:
            # Dispositivo (CPU/GPU) y método de simulación, ambos controlados
            # por variables de entorno (PCE_DEVICE / PCE_SIM_METHOD) definidas
            # en el job.sh del cluster correspondiente — ver
            # get_sim_device/get_sim_method. max_parallel_threads aprovecha
            # los cores reservados con --cpus-per-task (paraleliza el álgebra
            # lineal del statevector vía OpenMP; en QMIO es la única fuente
            # de paralelismo interno, en FT3 ayuda sobre todo a la
            # preparación de circuitos en CPU antes de mandarlos a la GPU).
            sim = AerSimulator(
                method=get_sim_method(),
                device=get_sim_device(),
                max_parallel_threads=get_max_parallel_threads(),
            )

        # --- Construir y transpilar UNA SOLA VEZ los tres circuitos de medida ---
        # (estructura + rotaciones de base + save_statevector/measure_all),
        # sin bind de parámetros todavía. Esto evita re-transpilar en cada
        # evaluación de la función de pérdida, que era el cuello de botella
        # principal.
        qc_z = base_circuit.copy()
        qc_x = base_circuit.copy()
        qc_y = base_circuit.copy()

        for q in range(qubits):
            qc_x.h(q)
        for q in range(qubits):
            qc_y.sdg(q)
            qc_y.h(q)

        if shots > 0:
            # Modo shots (incluye QMIO_REAL/QMIO_FAKE, que siempre pasan
            # por aquí): medir en base computacional (las rotaciones de
            # base ya están aplicadas arriba). measure_all() añade el
            # registro clásico necesario para poder pedir shots al run().
            qc_z.measure_all()
            qc_x.measure_all()
            qc_y.measure_all()
        else:
            # Modo exacto (comportamiento original, sin cambios). Solo
            # alcanzable con backend AER — QMIO_REAL/QMIO_FAKE ya lanzaron
            # ValueError más arriba si shots<=0.
            qc_z.save_statevector()
            qc_x.save_statevector()
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
                graph, list_size, qubits, d_t, experiment_result, shots,
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
        elif optimizer_lower == "rotosolve":
            # Optimizador libre de gradiente específico para puertas de
            # rotación de Pauli (ver docstring de rotosolve() más arriba).
            # No usa scipy: aquí self.pce_maxiter se interpreta como nº de
            # BARRIDOS completos sobre todos los parámetros, no como
            # "iteraciones" en el sentido de minimize/differential_evolution.
            rotosolve(loss, initial_params, maxiter=self.pce_maxiter)
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

    def _seed_for_instance(self, num_nodes: int, instance_index: int) -> int:
        """Semilla exacta de una instancia dada — misma fórmula que antes
        (self.seed + num_nodes*100_000 + instance_index), pero como
        función cerrada (no un contador que se incrementa en un bucle),
        para poder calcular la semilla de CUALQUIER instancia de forma
        aislada, sin depender de haber "recorrido" las anteriores. Esto es
        lo que hace posible trocear en chunks sin romper reproducibilidad:
        cada instancia es independientemente re-derivable por su índice."""
        return self.seed + num_nodes * 100_000 + instance_index

    def _run_instances_serial(self, num_nodes: int, instance_indices: List[int]) -> List[int]:
        """
        Calcula, EN SERIE, las instancias indicadas (por índice absoluto,
        no necesariamente 0..N-1 — puede ser un subconjunto, para el caso
        de chunks). Devuelve cut_sizes en el MISMO ORDEN que
        instance_indices. Incluye el log de progreso por instancia.
        """
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
                f"completada en {format_duration(instance_duration)} — "
                f"transcurrido={format_duration(elapsed)} "
                f"restante_estimado={format_duration(remaining_estimate)}"
            )
        return cut_sizes

    def _run_instances_parallel(self, num_nodes: int, instance_indices: List[int], num_workers: int) -> List[int]:
        """
        Igual que _run_instances_serial pero repartiendo las instancias
        entre num_workers procesos (multiprocessing.ProcessPoolExecutor),
        cada uno con una fracción de --cpus-per-task (get_max_parallel_threads()
        del proceso padre, dividido entre num_workers). Devuelve cut_sizes
        en el MISMO ORDEN que instance_indices, aunque internamente
        terminen en otro orden (multiproceso no garantiza orden de
        finalización).
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        cpus_per_task = get_max_parallel_threads()
        threads_per_worker = max(1, cpus_per_task // num_workers)
        base_seed = self.seed + num_nodes * 100_000

        qcvv_logger.info(
            f"[PCE] n={num_nodes} paralelizando {len(instance_indices)} instancias entre "
            f"{num_workers} workers ({threads_per_worker} hilos/worker, "
            f"{cpus_per_task} hilos totales disponibles)"
        )

        tasks = [
            (num_nodes, idx, base_seed, self.k, self.pce_maxiter, self.pce_optimizer,
             self.pce_alpha_factor, self.pce_beta, self.pce_de_popsize, self.pce_de_strategy,
             self.pce_de_mutation, self.pce_de_recombination, self.pce_de_tol, self.pce_de_polish,
             self.pce_de_init, threads_per_worker)
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
                    f"completada (paralelo) — transcurrido={format_duration(elapsed)}"
                )

        return [results_by_idx[idx] for idx in instance_indices]

    def execute_single_n(self, num_nodes: int) -> Dict:
        """
        Ejecuta el benchmark PCE para UN ÚNICO tamaño de grafo (num_nodes),
        con self.num_instances grafos aleatorios, EN SERIE (comportamiento
        original, sin chunks ni multiproceso). Pensado para lanzarse como
        una tarea independiente de un SLURM job array, en paralelo con
        otros tamaños de nodo — o usar execute_single_n_chunk() si además
        quieres repartir las instancias de un mismo n entre varias tareas.

        Devuelve un diccionario con los cortes obtenidos y el beta ya
        calculado, listo para loguear inmediatamente o guardar a disco.
        """
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
            f"[PCE] n={num_nodes} beta={approximation_ratio:.4f} ± {beta_std:.4f} ({status}) "
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
        """
        Ejecuta SOLO el subconjunto de instancias que le corresponde a
        este chunk (chunk_index de num_chunks totales), para un n dado.
        Pensado para lanzarse como una tarea SLURM independiente, en
        paralelo con los demás chunks de ese mismo n (varias tareas/nodos
        -> menor tiempo de pared por n), y opcionalmente paralelizando
        además las instancias DE ESTE CHUNK entre varios procesos locales
        con num_workers>1 (aprovechar varios cores del mismo nodo).

        Reparto de instancias por chunk: round-robin por índice
        (instancia i pertenece al chunk i % num_chunks) — determinista,
        no requiere coordinación entre tareas, y cada chunk recibe
        ceil o floor(num_instances/num_chunks) instancias, como mucho
        una de diferencia entre chunks.

        El beta/beta_std que devuelve este método son SOLO informativos
        de este chunk (útiles para ver que algo no está torcido a medio
        camino), NO el resultado final del n completo — eso lo calcula
        merge_chunks_qscore_PCE.py combinando los cut_sizes de TODOS los
        chunks de este n.

        Seguridad con hardware real: PCE_BACKEND=QMIO_REAL no admite
        ningún tipo de paralelismo (un solo chip físico, acceso síncrono),
        así que aquí se exige num_chunks=1 y num_workers=1 con ese
        backend — evita que alguien lance por error varias tareas
        peleándose por la cola de qpu a la vez.
        """
        if get_backend_name() == "QMIO_REAL" and (num_chunks > 1 or num_workers > 1):
            raise ValueError(
                "PCE_BACKEND=QMIO_REAL no admite paralelismo (ni num_chunks>1 ni "
                "num_workers>1) — un solo chip físico, acceso síncrono. "
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

        # beta/beta_std de ESTE chunk únicamente — informativo, no el
        # resultado final (compute_beta necesita al menos 1 elemento;
        # con instance_indices vacío (num_chunks > num_instances) no
        # debería llegar aquí porque el lanzador ya evita esa situación).
        chunk_beta, chunk_beta_std = compute_beta(cut_sizes, num_nodes)

        shots = get_shots()
        backend_name = get_backend_name()
        qcvv_logger.info(
            f"[PCE] n={num_nodes} chunk {chunk_index + 1}/{num_chunks} completado — "
            f"beta_parcial={chunk_beta:.4f} duration={format_duration(duration_seconds)} "
            f"shots={shots if shots > 0 else 'exact'} backend={backend_name}"
        )

        return {
            "num_nodes": num_nodes,
            "num_qubits": qubits,
            "num_instances": self.num_instances,   # TOTAL del n, no de este chunk
            "chunk_index": chunk_index,
            "num_chunks": num_chunks,
            "num_workers": num_workers,
            "instance_indices": instance_indices,
            "cut_sizes": cut_sizes,                # SOLO las de este chunk, alineadas con instance_indices
            "beta_chunk": chunk_beta,               # informativo, NO es el beta final del n
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
        """Guarda el resultado de execute_single_n en <outdir>/n_<num_nodes>.json."""
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"n_{result['num_nodes']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return path

    @staticmethod
    def save_result_hdf5(result: Dict, outdir: str, filename: str = None) -> str:
        """
        Guarda el resultado de execute_single_n / execute_single_n_chunk en
        <outdir>/<filename>, o <outdir>/n_<num_nodes>.h5 si no se indica
        filename (comportamiento por defecto, igual que antes).

        Estructura del fichero:
        - Dataset 'cut_sizes': array con los cortes de cada instancia.
        - Atributos (metadatos) a nivel raíz: todos los demás campos del
          dict (num_nodes, num_qubits, num_instances, beta, beta_std,
          success, timestamp, duration_seconds, k, seed, pce_optimizer,
          pce_maxiter, simulator, device) — incluye todo lo que aparece
          en el título de las gráficas más el simulador/dispositivo usado.

        filename explícito: usado por los ficheros de chunk
        (n_<N>_chunk_<i>_of_<num_chunks>.h5) para que el agregador de
        siempre (glob 'n_*.h5' sobre num_nodes) no los confunda con
        resultados completos — los chunks se guardan en un subdirectorio
        separado (ver run_qscore_PCE_chunk.py) y solo el fusionador de
        chunks (merge_chunks_qscore_PCE.py) escribe el n_<N>.h5 final ahí.

        Sustituye a save_result_json para los runs nuevos; el agregador
        sigue sabiendo leer los .json de runs antiguos.
        """
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
                    # h5py no admite None como atributo (TypeError: Object
                    # dtype... no native HDF5 equivalent). Ocurre sobre todo
                    # con device_actual cuando num_workers>1: los metadatos
                    # de Aer se detectan DENTRO de los procesos hijo, así que
                    # el proceso padre nunca llega a rellenar esa variable.
                    # Se omite el atributo entero en vez de escribir un
                    # placeholder — el resto del código ya usa
                    # result.get("device_actual") con valor por defecto, así
                    # que la ausencia del atributo se maneja igual de bien
                    # que si valiera None.
                    continue
                f.attrs[key] = value
        return path

    def execute(self) -> BenchmarkRunResult:
        """
        Genera los grafos y ejecuta PCE para cada tamaño/instancia EN SERIE.
        Se mantiene por compatibilidad / pruebas rápidas en un solo proceso;
        para paralelizar por n usa execute_single_n() desde un job array.
        """
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
            # Reconstruir graph_list solo para mantener compatibilidad de formato
            # (execute_single_n no los devuelve para no duplicar memoria innecesariamente)
            dataset.attrs[num_nodes] = {
                "cut_sizes": result["cut_sizes"],
                "num_qubits": result["num_qubits"],
                "duration_seconds": result["duration_seconds"],
            }

        return BenchmarkRunResult(dataset=dataset)


def qscore_pce_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Misma métrica beta que en qscore.py, aplicada a los cortes obtenidos con PCE."""
    import matplotlib.pyplot as plt

    plots = {}
    observations: List[BenchmarkObservation] = []
    dataset = run.dataset.copy(deep=True)

    simulator = dataset.attrs.get("simulator")
    device = dataset.attrs.get("device")
    shots = dataset.attrs.get("shots", 0)  # retrocompatible: runs antiguos sin este campo = modo exacto
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
                f"[PCE] n={num_nodes} PASSED | beta={approximation_ratio:.4f} ± {std_of_approximation_ratio:.4f}{qubits_str}{duration_str}"
            )
            qscore = num_nodes
        else:
            qcvv_logger.info(
                f"[PCE] n={num_nodes} FAILED | beta={approximation_ratio:.4f} ± {std_of_approximation_ratio:.4f} < 0.2{qubits_str}{duration_str}"
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

    # --- Líneas verticales en los saltos de número de qubits ---
    # Solo se marca el primer n en el que se empieza a usar un qubit más
    # (no se repite la línea en cada n posterior que siga usando ese mismo número).
    if all(q is not None for q in qubits_list):
        # El primer punto no tiene "salto" que marcar (no hay n anterior con
        # el que compararlo), pero sin indicar su valor de partida no se puede
        # saber si un salto posterior fue de +1 o de +varios qubits de golpe.
        # Por eso se anota aparte, sin línea vertical (no es un salto, es el inicio).
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
    # En modo shots el método de Aer (statevector/mps/...) es un detalle de
    # implementación interno, no lo relevante para el lector de la gráfica
    # — lo relevante es que la medida viene de un muestreo con N shots, no
    # de leer el estado exacto. Por eso es un if/else excluyente: se muestra
    # el simulador SOLO en modo exacto, y los shots SOLO en modo shots,
    # nunca los dos a la vez.
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