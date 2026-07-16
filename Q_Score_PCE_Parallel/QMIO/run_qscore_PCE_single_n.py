"""
run_qscore_PCE_single_n.py
===========================
Ejecuta el benchmark Q-Score-PCE para UN ÚNICO tamaño de grafo (--num_nodes).
Pensado para lanzarse como una tarea de un SLURM job array, en paralelo con
otros tamaños de nodo. Cada tarea guarda su resultado en un JSON individual
dentro de --outdir; el agregador (aggregate_qscore_PCE.py) los junta después.

Uso:
    python run_qscore_PCE_single_n.py --num_nodes 14 --num_instances 50 \
        --k 2 --pce_maxiter 10 --pce_optimizer differentialevolution \
        --outdir Resultados/PCE
"""

import argparse

from qscore_PCE import QScoreBenchmarkPCE, QScoreConfiguration, get_shots


def parse_args():
    parser = argparse.ArgumentParser(description="Q-Score-PCE para un único n")
    parser.add_argument("--num_nodes", type=int, required=True, help="Número de nodos del grafo")
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--pce_maxiter", type=int, default=10)
    parser.add_argument("--pce_optimizer", type=str, default="COBYLA")
    parser.add_argument("--pce_alpha_factor", type=float, default=1.5)
    parser.add_argument("--pce_beta", type=float, default=0.5)
    parser.add_argument("--pce_de_popsize", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="Resultados/PCE")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n==============================")
    print(f"num_nodes    = {args.num_nodes}")
    print(f"num_instances= {args.num_instances}")
    print(f"k            = {args.k}")
    print(f"optimizer    = {args.pce_optimizer}")
    print(f"maxiter      = {args.pce_maxiter}")
    print(f"outdir       = {args.outdir}")
    print(f"shots        = {get_shots() or 'exacto (statevector)'}")
    print("==============================\n")

    # min/max/step no se usan en execute_single_n, pero QScoreConfiguration
    # los requiere: los fijamos a num_nodes para que quede coherente si se
    # inspecciona la config guardada.
    config = QScoreConfiguration(
        num_instances=args.num_instances,
        min_num_nodes=args.num_nodes,
        max_num_nodes=args.num_nodes,
        node_step=1,
        seed=args.seed,
        k=args.k,
        pce_maxiter=args.pce_maxiter,
        pce_optimizer=args.pce_optimizer,
        pce_alpha_factor=args.pce_alpha_factor,
        pce_beta=args.pce_beta,
        pce_de_popsize=args.pce_de_popsize,
    )

    benchmark = QScoreBenchmarkPCE(configuration=config)
    result = benchmark.execute_single_n(args.num_nodes)
    path = benchmark.save_result_hdf5(result, args.outdir)

    result_shots = result.get("shots", 0)
    print(f"\n✔ n={args.num_nodes} beta={result['beta']:.4f} ± {result['beta_std']:.4f} "
          f"({'PASSED' if result['success'] else 'FAILED'}) "
          f"shots={result_shots if result_shots else 'exact'}")
    print(f"✔ Resultado guardado en: {path}")


if __name__ == "__main__":
    main()