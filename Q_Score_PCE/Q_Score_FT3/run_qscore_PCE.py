from Pruebas_iniciales.qscore_PCE import QScoreBenchmarkPCE, QScoreConfiguration, qscore_pce_analysis

config = QScoreConfiguration(
    num_instances=50,
    min_num_nodes=4,
    max_num_nodes=64,
    node_step=10,
    seed=42,
    k=2,                  # orden de compresión PCE
    pce_maxiter=30,       # pocas iteraciones para la primera prueba
    pce_optimizer="DIFFERENTIALEVOLUTION",
    pce_alpha_factor=1.5, # aplha = pce_alpha_factor * num_qubits
    pce_beta=0.5,
)

benchmark = QScoreBenchmarkPCE(configuration=config)
run_result = benchmark.execute()
analysis = qscore_pce_analysis(run_result)

for obs in analysis.observations:
    print(f"{obs.name}: {obs.value}")

for fig_name, fig in analysis.plots.items():
    fig.savefig(fig_name)
    print(f"Figura guardada: {fig_name}")