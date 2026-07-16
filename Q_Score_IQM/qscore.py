# pylint: disable=too-many-lines
"""
Versión adaptada para el QMIO de Q-Score a partir del código 
del repositorio de IQM para un QAOA. 

Con ayuda de Claude.ia

Eliminadas las dependencias de iqm.benchmarks.* y reemplazadas
por implementaciones directas con Qiskit y librerías estándar.
"""

import itertools
import logging
from dataclasses import dataclass, field
from time import strftime
from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from networkx import Graph
import networkx as nx
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZZGate
from qiskit.compiler import transpile
from scipy.optimize import basinhopping, minimize
import xarray as xr

# ---------------------------------------------------------------------------
# Reemplazos del framework iqm.benchmarks
# ---------------------------------------------------------------------------

# Logger estándar en lugar de qcvv_logger
qcvv_logger = logging.getLogger("qscore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class QScoreConfiguration:
    """Configuración del benchmark Q-Score (versión standalone)."""
    num_instances: int = 10
    min_num_nodes: int = 2
    max_num_nodes: int = 5
    num_qaoa_layers: int = 1
    use_virtual_node: bool = False
    use_classically_optimized_angles: bool = False
    choose_qubits_routine: str = "naive"
    qiskit_optim_level: int = 1
    optimize_sqg: bool = False
    REM: bool = False
    mit_shots: int = 1000
    seed: int = 42
    num_trials: int = 1
    num_shots: int = 1000
    custom_qubits_array: Optional[List[List[int]]] = None

    def __iter__(self):
        """Permite iterar sobre la configuración como dict (usado en add_all_meta_to_dataset)."""
        for k, v in self.__dict__.items():
            yield k, v


@dataclass
class BenchmarkRunResult:
    """Contenedor simple para los resultados de ejecución."""
    dataset: xr.Dataset


@dataclass
class BenchmarkObservation:
    """Observación individual del benchmark."""
    name: str
    value: object
    uncertainty: float = 0.0
    identifier: object = None


@dataclass
class BenchmarkAnalysisResult:
    """Resultado del análisis del benchmark."""
    dataset: xr.Dataset
    plots: Dict
    observations: List[BenchmarkObservation]


def add_counts_to_dataset(dataset: xr.Dataset, counts: Dict[str, int], key: str):
    """Almacena counts en el dataset como atributo."""
    dataset.attrs[key] = counts


def apply_readout_error_mitigation(counts, *args, **kwargs):
    """Stub: REM desactivado, devuelve los counts sin modificar."""
    return counts


# ---------------------------------------------------------------------------
# Funciones de utilidad que reemplazan iqm.benchmarks.utils
# ---------------------------------------------------------------------------

def perform_backend_transpilation(circuit: QuantumCircuit, backend, qubits=None,
                                   coupling_map=None, qiskit_optim_level=1,
                                   optimize_sqg=False, **kwargs) -> QuantumCircuit:
    """Transpila un circuito al backend usando Qiskit estándar."""
    return transpile(
        circuit,
        backend=backend,
        optimization_level=qiskit_optim_level,
        initial_layout=qubits,
    )


def submit_execute(circuits: List[QuantumCircuit], backend, num_shots: int = 1000,
                   **kwargs) -> object:
    """Ejecuta una lista de circuitos en el backend y devuelve el job."""
    job = backend.run(circuits, shots=num_shots)
    return job


def retrieve_all_counts(job, num_circuits: int) -> List[Dict[str, int]]:
    """Recupera los counts de todos los circuitos de un job."""
    result = job.result()
    counts_list = []
    for i in range(num_circuits):
        try:
            counts_list.append(result.get_counts(i))
        except Exception:
            counts_list.append(result.get_counts())
    return counts_list


def xrvariable_to_counts(dataset: xr.Dataset, num_nodes: int,
                          num_instances: int) -> List[Dict[str, int]]:
    """Extrae los counts almacenados en el dataset para un tamaño de grafo dado."""
    counts_list = []
    for inst_idx in range(num_instances):
        key = f"counts_{num_nodes}_{inst_idx}"
        counts_list.append(dataset.attrs.get(key, {}))
    return counts_list


# ---------------------------------------------------------------------------
# Funciones principales del Q-Score (sin modificar respecto al original)
# ---------------------------------------------------------------------------

def calculate_optimal_angles_for_QAOA_p1(graph: Graph) -> List[float]:
    """Calculates the optimal angles for single layer QAOA MaxCut ansatz."""

    def get_Zij_maxcut_p1(edge_ij, gamma, beta):
        i, j = edge_ij
        di = graph.degree[i]
        dj = graph.degree[j]
        first = np.cos(2 * gamma) ** (di - 1) + np.cos(2 * gamma) ** (dj - 1)
        first *= 0.5 * np.sin(4 * beta) * np.sin(2 * gamma)
        node_list = list(graph.nodes).copy()
        node_list.remove(i)
        node_list.remove(j)
        f1 = 1
        f2 = 1
        for k in node_list:
            if graph.has_edge(i, k) and graph.has_edge(j, k):
                f1 *= np.cos(4 * gamma)
            elif graph.has_edge(i, k) or graph.has_edge(j, k):
                f1 *= np.cos(2 * gamma)
            f2 *= np.cos(2 * gamma)
        second = 0.5 * np.sin(2 * beta) ** 2 * (f1 - f2)
        return first - second

    def get_expected_zz_edgedensity(x):
        gamma = x[0]
        beta = x[1]
        return sum([get_Zij_maxcut_p1(edge, gamma, beta) for edge in graph.edges]) / graph.number_of_edges()

    bounds = [(0.0, np.pi / 2), (-np.pi / 4, 0.0)]
    x_init = [0.15, -0.28]
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(get_expected_zz_edgedensity, x_init, minimizer_kwargs=minimizer_kwargs, niter=10, T=2)
    return list(res.x)


def cut_cost_function(x: str, graph: Graph) -> int:
    """Returns the number of cut edges (with minus sign)."""
    obj = 0
    for i, j in graph.edges():
        if x[i] != x[j]:
            obj += 1
    return -1 * obj


def compute_expectation_value(
    counts: Dict[str, int], graph: Graph, qubit_to_node: Dict[int, int],
    virtual_nodes: List[Tuple[int, int]]
) -> float:
    """Computes expectation value based on measurement results."""
    avg = 0
    sum_count = 0
    num_logical_qubits= len(qubit_to_node)

    for bitstring_aux, count in counts.items():
        bits = list(bitstring_aux)[::-1]
        bits = bits[:num_logical_qubits]

        num_nodes = graph.number_of_nodes()
        bitstring = ["0"] * num_nodes

        for qubit, node in qubit_to_node.items():
            if qubit < len(bits) and node < num_nodes:
                bitstring[node] = bits[qubit]

        for virtual_node in virtual_nodes:
            if virtual_node[0] is not None and virtual_node[0] < num_nodes:
                bitstring[virtual_node[0]] = str(virtual_node[1])

        joined = "".join(bitstring)
        if len(joined) != graph.number_of_nodes():
            print(f"DEBUG: bitstring='{joined}' len={len(joined)}, "
                f"graph_nodes={graph.number_of_nodes()}, "
                f"qubit_to_node={qubit_to_node}, "
                f"bits_len={len(bits)}, "
                f"bitstring_aux='{bitstring_aux}'")
        obj = cut_cost_function(joined, graph)
        avg += obj * count
        sum_count += count
    return avg / sum_count


def create_objective_function(
    counts: Dict[str, int], graph: Graph, qubit_to_node: Dict[int, int],
    virtual_nodes: List[Tuple[int, int]]
) -> Callable:
    """Creates the objective function for QAOA optimization."""
    def objective_function(temp):
        return compute_expectation_value(counts, graph, qubit_to_node, virtual_nodes)
    return objective_function


def is_successful(approximation_ratio: float) -> bool:
    """Check whether approximation ratio is above beta* = 0.2 threshold."""
    return bool(approximation_ratio > 0.2)


def get_optimal_angles(num_layers: int) -> List[float]:
    """Provides optimal angles for QAOA MaxCut ansatz (Wurtz et al. 2021)."""
    OPTIMAL_INITIAL_ANGLES = {
        "1": [-0.616, 0.393 / 2],
        "2": [-0.488, 0.898 / 2, 0.555 / 2, 0.293 / 2],
        "3": [-0.422, 0.798 / 2, 0.937 / 2, 0.609 / 2, 0.459 / 2, 0.235 / 2],
        "4": [-0.409, 0.781 / 2, 0.988 / 2, 1.156 / 2, 0.600 / 2, 0.434 / 2, 0.297 / 2, 0.159 / 2],
        "5": [-0.36, -0.707, -0.823, -1.005, -1.154, 0.632 / 2, 0.523 / 2, 0.390 / 2, 0.275 / 2, 0.149 / 2],
    }
    if num_layers > 5:
        raise ValueError("Este standalone soporta hasta 5 capas QAOA")
    return OPTIMAL_INITIAL_ANGLES[str(num_layers)]


def run_QAOA(
    counts: Dict[str, int],
    graph_physical: Graph,
    qubit_node: Dict[int, int],
    use_classical_angles: bool,
    qaoa_layers: int,
    virtual_nodes: List[Tuple[int, int]],
) -> float:
    """Solves the cut size of MaxCut for a graph using QAOA."""
    objective_function = create_objective_function(counts, graph_physical, qubit_node, virtual_nodes)
    if use_classical_angles:
        if graph_physical.number_of_edges() != 0:
            opt_angles = calculate_optimal_angles_for_QAOA_p1(graph_physical)
        else:
            opt_angles = [1.0, 1.0]
        res = minimize(objective_function, opt_angles, method="COBYLA", tol=1e-5, options={"maxiter": 0})
    else:
        theta = get_optimal_angles(qaoa_layers)
        bounds = [(-np.pi, np.pi)] * qaoa_layers + [(0.0, np.pi)] * qaoa_layers
        res = minimize(objective_function, theta, bounds=bounds, method="COBYLA",
                       tol=1e-5, options={"maxiter": 300})
    return -res.fun


def qscore_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a QScore experiment."""
    plots = {}
    observations: List[BenchmarkObservation] = []
    dataset = run.dataset.copy(deep=True)

    backend_name = dataset.attrs["backend_name"]
    timestamp = dataset.attrs["execution_timestamp"]
    nodes_list = dataset.attrs["node_numbers"]
    num_instances: int = dataset.attrs["num_instances"]
    use_virtual_node: bool = dataset.attrs["use_virtual_node"]
    use_classically_optimized_angles = dataset.attrs["use_classically_optimized_angles"]
    num_qaoa_layers = dataset.attrs["num_qaoa_layers"]

    qscore = 0
    beta_ratio_list = []
    beta_ratio_std_list = []

    for num_nodes in nodes_list:
        dataset_dictionary = dataset.attrs[num_nodes]
        graph_list = dataset_dictionary["graph"]
        qubit_to_node_list = dataset_dictionary["qubit_to_node"]
        virtual_node_list = dataset_dictionary["virtual_nodes"]
        no_edge_instances = dataset_dictionary["no_edge_instances"]

        cut_sizes_list = [0.0] * len(no_edge_instances)
        instances_with_edges = set(range(num_instances)) - set(no_edge_instances)
        num_instances_with_edges = len(instances_with_edges)

        execution_results = xrvariable_to_counts(dataset, num_nodes, num_instances_with_edges)

        for inst_idx, instance in enumerate(list(instances_with_edges)):
            cut_sizes = run_QAOA(
                execution_results[inst_idx],
                graph_list[instance],
                qubit_to_node_list[instance],
                use_classically_optimized_angles,
                num_qaoa_layers,
                virtual_node_list[instance],
            )
            cut_sizes_list.append(cut_sizes)

        LAMBDA = 0.178
        average_cut_size = np.mean(cut_sizes_list) - num_nodes * (num_nodes - 1) / 8
        average_best_cut_size = LAMBDA * pow(num_nodes, 3 / 2)
        approximation_ratio = float(average_cut_size / average_best_cut_size)

        approximation_ratio_list = [
            (np.array(cut_sizes) - num_nodes * (num_nodes - 1) / 8) / (LAMBDA * num_nodes ** (3 / 2))
            for cut_sizes in cut_sizes_list
        ]

        beta_ratio_list.append(np.mean(approximation_ratio_list))
        success = is_successful(approximation_ratio)
        std_of_approximation_ratio = np.std(np.array(approximation_ratio_list)) / np.sqrt(
            len(approximation_ratio_list) - 1
        )
        beta_ratio_std_list.append(std_of_approximation_ratio)

        if success:
            qcvv_logger.info(
                f"Q-Score = {num_nodes} PASSED | beta = {approximation_ratio:.4f} ± {std_of_approximation_ratio:.4f} | Avg cut: {np.mean(cut_sizes_list):.4f}"
            )
            qscore = num_nodes
        else:
            qcvv_logger.info(
                f"Q-Score = {num_nodes} FAILED | beta = {approximation_ratio:.4f} ± {std_of_approximation_ratio:.4f} < 0.2 | Avg cut: {np.mean(cut_sizes_list):.4f}"
            )

        observations.extend([
            BenchmarkObservation(name="mean_approximation_ratio", value=approximation_ratio,
                                 uncertainty=std_of_approximation_ratio),
            BenchmarkObservation(name="is_successful", value=str(success)),
            BenchmarkObservation(name="Qscore_result", value=qscore if success else 1),
        ])

        dataset.attrs[num_nodes].update({"approximate_ratio_list": approximation_ratio_list})

    fig_name, fig = plot_approximation_ratios(
        nodes_list, beta_ratio_list, beta_ratio_std_list,
        use_virtual_node, use_classically_optimized_angles,
        num_instances, backend_name, timestamp,
    )
    plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)


def plot_approximation_ratios(
    nodes, beta_ratio, beta_std, use_virtual_node,
    use_classically_optimized_angles, num_instances, backend_name, timestamp
) -> tuple:
    """Generate the figure of approximation ratios vs number of nodes."""
    fig = plt.figure()
    ax = plt.axes()
    plt.axhline(0.2, color="red", linestyle="dashed", label="Threshold")
    plt.errorbar(nodes, beta_ratio, yerr=beta_std, fmt="-o", capsize=10,
                 markersize=8, color="#759DEB", label="Approximation ratio")
    ax.set_ylabel(r"Q-score ratio $\beta(n)$")
    ax.set_xlabel("Number of nodes $(n)$")
    plt.xticks(range(min(nodes), max(nodes) + 1))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title(f"Q-score, {num_instances} instances\nBackend: {backend_name} / {timestamp}", fontsize=9)
    fig_name = f"{max(nodes)}_nodes_{num_instances}_instances.png"
    plt.gcf().set_dpi(250)
    plt.close()
    return fig_name, fig


# ---------------------------------------------------------------------------
# Clase principal QScoreBenchmark (versión standalone)
# ---------------------------------------------------------------------------

class QScoreBenchmark:
    """
    Q-score benchmark — versión standalone para el QMIO.
    Elimina dependencias de iqm.benchmarks.* y usa Qiskit directamente.
    """

    def __init__(self, backend_arg, configuration: QScoreConfiguration):
        self.backend = backend_arg
        self.configuration = configuration
        self.backend_configuration_name = getattr(backend_arg, "name", str(backend_arg))

        self.num_instances = configuration.num_instances
        self.num_qaoa_layers = configuration.num_qaoa_layers
        self.min_num_nodes = configuration.min_num_nodes
        self.max_num_nodes = configuration.max_num_nodes
        self.use_virtual_node = configuration.use_virtual_node
        self.use_classically_optimized_angles = configuration.use_classically_optimized_angles
        self.choose_qubits_routine = configuration.choose_qubits_routine
        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.optimize_sqg = configuration.optimize_sqg
        self.REM = configuration.REM
        self.mit_shots = configuration.mit_shots
        self.num_shots = configuration.num_shots
        self.seed = configuration.seed
        self.num_trials = configuration.num_trials

        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

        self.graph_physical: Graph = nx.Graph()
        self.virtual_nodes: List[Tuple[int, int]] = []
        self.node_to_qubit: Dict[int, int] = {}
        self.qubit_to_node: Dict[int, int] = {}

        if self.use_classically_optimized_angles and self.num_qaoa_layers > 1:
            raise ValueError("use_classically_optimized_angles requiere num_qaoa_layers=1.")
        if self.use_virtual_node and self.num_qaoa_layers > 1:
            raise ValueError("use_virtual_node requiere num_qaoa_layers=1.")

        if self.choose_qubits_routine == "custom":
            self.custom_qubits_array = [
                list(x) for x in cast(Sequence[Sequence[int]], configuration.custom_qubits_array)
            ]

    @staticmethod
    def choose_qubits_naive(num_qubits: int) -> List[int]:
        if num_qubits == 2:
            return [0, 2]
        return list(range(num_qubits))

    def choose_qubits_custom(self, num_qubits: int) -> List[int]:
        selected_qubits = [q for q in self.custom_qubits_array if len(q) == num_qubits]
        if len(selected_qubits) > 1:
            chosen_qubits = selected_qubits[0]
            self.custom_qubits_array.remove(chosen_qubits)
        else:
            chosen_qubits = selected_qubits[0]
        return list(chosen_qubits)

    def generate_maxcut_ansatz(
        self, graph: Graph, theta: List[float], rzz_list=None
    ) -> QuantumCircuit:
        """Generate QAOA MaxCut ansatz circuit."""
        gamma = theta[: self.num_qaoa_layers]
        beta = theta[self.num_qaoa_layers :]

        if self.graph_physical.number_of_nodes() != graph.number_of_nodes():
            num_qubits = self.graph_physical.number_of_nodes()
            self.node_to_qubit = {node: qubit for qubit, node in enumerate(list(self.graph_physical.nodes))}
            self.qubit_to_node = dict(enumerate(list(self.graph_physical.nodes)))
        else:
            num_qubits = graph.number_of_nodes()
            self.node_to_qubit = {node: node for node in list(self.graph_physical.nodes)}
            self.qubit_to_node = self.node_to_qubit.copy()

        if num_qubits == 0:
            return QuantumCircuit(1)

        qaoa_qc = QuantumCircuit(num_qubits)

        for i in range(num_qubits):
            qaoa_qc.h(i)

        for layer in range(self.num_qaoa_layers):
            if rzz_list is not None and layer == 0:
                for rzzs in rzz_list:
                    qaoa_qc.rzz(2 * gamma[layer], rzzs[0], rzzs[1])
            else:
                for edge in self.graph_physical.edges():
                    i = self.node_to_qubit[edge[0]]
                    j = self.node_to_qubit[edge[1]]
                    qaoa_qc.rzz(2 * gamma[layer], i, j)

            for vn in self.virtual_nodes:
                for edge in graph.edges(vn[0]):
                    edges_between_virtual_nodes = list(
                        itertools.combinations([i[0] for i in self.virtual_nodes], 2)
                    )
                    if set(edge) not in list(map(set, edges_between_virtual_nodes)):
                        sign = -1.0 if vn[1] == 1 else 1.0
                        qaoa_qc.rz(sign * 2.0 * gamma[layer], self.node_to_qubit[edge[1]])

            for i in range(num_qubits):
                qaoa_qc.rx(2 * beta[layer], i)

        qaoa_qc.measure_all()
        return qaoa_qc

    def execute(self, backend) -> BenchmarkRunResult:
        """Executes the Q-Score benchmark."""
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")
        dataset = xr.Dataset()

        # Metadata
        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend_name"] = getattr(backend, "name", "FakeQmio")
        dataset.attrs["backend_configuration_name"] = self.backend_configuration_name
        for key, value in self.configuration:
            dataset.attrs[key] = value

        if self.use_virtual_node:
            max_num_nodes = self.max_num_nodes + 1
        else:
            max_num_nodes = self.max_num_nodes

        node_numbers = list(range(self.min_num_nodes, max_num_nodes + 1))
        dataset.attrs["max_num_nodes"] = node_numbers[-1]
        dataset.attrs["node_numbers"] = node_numbers

        for num_nodes in node_numbers:
            if self.use_virtual_node:
                updated_num_nodes = num_nodes - 1
            else:
                updated_num_nodes = num_nodes

            qcvv_logger.info(f"Ejecutando {self.num_instances} grafos aleatorios con {num_nodes} nodos.")

            if self.choose_qubits_routine.lower() == "naive":
                qubit_set = self.choose_qubits_naive(updated_num_nodes)
            elif self.choose_qubits_routine.lower() == "custom":
                qubit_set = self.choose_qubits_custom(updated_num_nodes)
            else:
                raise ValueError('choose_qubits_routine debe ser "naive" o "custom".')

            seed = self.seed
            graph_list = []
            qubit_to_node_list = []
            virtual_node_list = []
            no_edge_instances = []
            qc_list = []

            for instance in range(self.num_instances):
                # Generar grafo aleatorio Erdos-Renyi con densidad 0.5
                graph = nx.erdos_renyi_graph(num_nodes, 0.5, seed=seed)
                seed += 1
                graph_list.append(graph)

                # Nodo virtual
                if self.use_virtual_node:
                    virtual_node_idx = num_nodes - 1
                    virtual_node_val = np.random.randint(0, 2)
                    self.virtual_nodes = [(virtual_node_idx, virtual_node_val)]
                    self.graph_physical = graph.subgraph(
                        [n for n in graph.nodes if n != virtual_node_idx]
                    ).copy()
                else:
                    self.virtual_nodes = []
                    self.graph_physical = graph.copy()

                virtual_node_list.append(self.virtual_nodes)

                if self.graph_physical.number_of_edges() == 0:
                    no_edge_instances.append(instance)
                    qc_list.append(None)
                    qubit_to_node_list.append({})
                    continue

                # Ángulos iniciales
                if self.use_classically_optimized_angles:
                    theta = calculate_optimal_angles_for_QAOA_p1(self.graph_physical)
                else:
                    theta = get_optimal_angles(self.num_qaoa_layers)

                # Generar circuito
                qc = self.generate_maxcut_ansatz(graph, theta)

                # Transpilar
                qc_list.append(qc)
                qubit_to_node_list.append(self.qubit_to_node.copy())

            circuits_to_run_untranspiled = [qc for qc in qc_list if qc is not None]

            if circuits_to_run_untranspiled:
                qc_transpiled_list = transpile(
                    circuits_to_run_untranspiled,
                    backend=backend,
                    initial_layout=qubit_set,
                    optimization_level=self.qiskit_optim_level,
                )
            else:
                qc_transpiled_list = []

            # Luego ejecuta los ya transpilados:
            if qc_transpiled_list:
                job = submit_execute(qc_transpiled_list, backend, num_shots=self.num_shots)
                counts_list = retrieve_all_counts(job, len(qc_transpiled_list))
            else:
                counts_list = []

            # Ejecutar solo los circuitos con aristas
            circuits_to_run = [qc for qc in qc_list if qc is not None]
            instances_with_edges = [i for i in range(self.num_instances) if i not in no_edge_instances]

            if circuits_to_run:
                job = submit_execute(qc_transpiled_list, backend, num_shots=self.num_shots)
                counts_list = retrieve_all_counts(job, len(qc_transpiled_list))
            else:
                counts_list = []

            # Guardar counts en dataset
            for idx, instance in enumerate(instances_with_edges):
                key = f"counts_{num_nodes}_{idx}"
                dataset.attrs[key] = counts_list[idx]

            dataset.attrs[num_nodes] = {
                "graph": graph_list,
                "qubit_set": [qubit_set],
                "qubit_to_node": qubit_to_node_list,
                "virtual_nodes": virtual_node_list,
                "no_edge_instances": no_edge_instances,
            }

        return BenchmarkRunResult(dataset=dataset)