from Q_Score.qscore import QScoreBenchmark, QScoreConfiguration, qscore_analysis
from qmiotools.integrations.qiskitqmio import FakeQmio

backend = FakeQmio(thermal_relaxation=False)

config = QScoreConfiguration(
    num_instances=50,
    min_num_nodes=2,
    max_num_nodes=8,
    num_qaoa_layers=2,
    use_virtual_node=False,
    use_classically_optimized_angles=False,
    choose_qubits_routine="naive",
    qiskit_optim_level=1,
    optimize_sqg=False,
    REM=False,
    mit_shots=1000,
    num_shots=1000,
    seed=42,
    num_trials=1,
)

benchmark = QScoreBenchmark(backend_arg=backend, configuration=config)
run_result = benchmark.execute(backend)
analysis = qscore_analysis(run_result)

for obs in analysis.observations:
    print(f"{obs.name}: {obs.value}")

for fig_name, fig in analysis.plots.items():
    fig.savefig(fig_name)
    print(f"Figura guardada: {fig_name}")