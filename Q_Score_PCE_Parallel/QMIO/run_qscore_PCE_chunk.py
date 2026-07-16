"""
run_qscore_PCE_chunk.py
========================
Ejecuta un ÚNICO CHUNK de instancias para un tamaño de grafo (--num_nodes).
Pensado para lanzarse como una tarea de un SLURM job array, en paralelo con
otros chunks del MISMO n (y con chunks de otros n) — Opción B del reparto
por instancias. Dentro de cada chunk, además, las instancias pueden
repartirse entre --num_workers procesos locales (Opción A).

Guarda su resultado PARCIAL en un fichero de chunk (NO en el n_<N>.h5
estándar, para no colisionar con el agregador normal) — hace falta
merge_chunks_qscore_PCE.py para combinar todos los chunks de un n en el
n_<N>.h5 final antes de poder agregarlos con aggregate_qscore_PCE.py.

Uso:
    python run_qscore_PCE_chunk.py --num_nodes 500 --num_instances 30 \
        --chunk_index 0 --num_chunks 6 --num_workers 4 \
        --k 2 --pce_maxiter 15 --pce_optimizer differentialevolution \
        --outdir Resultados/PCE/<RUN_ID>/chunks
"""

import argparse

from qscore_PCE import QScoreBenchmarkPCE, QScoreConfiguration, get_backend_name


def parse_args():
    parser = argparse.ArgumentParser(description="Q-Score-PCE — un chunk de instancias para un único n")
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_instances", type=int, default=50, help="Nº TOTAL de instancias del n (no de este chunk)")
    parser.add_argument("--chunk_index", type=int, required=True, help="Índice de este chunk (0-indexado)")
    parser.add_argument("--num_chunks", type=int, required=True, help="Nº total de chunks en los que se divide este n")
    parser.add_argument("--num_workers", type=int, default=1, help="Procesos en paralelo dentro de este chunk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--pce_maxiter", type=int, default=10)
    parser.add_argument("--pce_optimizer", type=str, default="COBYLA")
    parser.add_argument("--pce_alpha_factor", type=float, default=1.5)
    parser.add_argument("--pce_beta", type=float, default=0.5)
    parser.add_argument("--pce_de_popsize", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="Resultados/PCE/chunks")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
        raise ValueError(f"chunk_index={args.chunk_index} fuera de rango [0, {args.num_chunks - 1}]")
    if args.num_chunks > args.num_instances:
        raise ValueError(
            f"num_chunks={args.num_chunks} > num_instances={args.num_instances}: "
            "habría chunks sin ninguna instancia asignada."
        )

    print("\n==============================")
    print(f"num_nodes     = {args.num_nodes}")
    print(f"num_instances = {args.num_instances} (total del n)")
    print(f"chunk         = {args.chunk_index + 1}/{args.num_chunks}")
    print(f"num_workers   = {args.num_workers}")
    print(f"k             = {args.k}")
    print(f"optimizer     = {args.pce_optimizer}")
    print(f"maxiter       = {args.pce_maxiter}")
    print(f"backend       = {get_backend_name()}")
    print(f"outdir        = {args.outdir}")
    print("==============================\n")

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

    # Nº de instancias que le van a tocar a ESTE chunk concreto (mismo
    # reparto round-robin que hace execute_single_n_chunk internamente) —
    # capar num_workers a ese número evita pedir más procesos de los que
    # habría trabajo que repartirles (p.ej. un chunk de 2 instancias no
    # necesita 8 workers).
    instances_in_this_chunk = len([i for i in range(args.num_instances) if i % args.num_chunks == args.chunk_index])
    effective_workers = max(1, min(args.num_workers, instances_in_this_chunk))
    if effective_workers != args.num_workers:
        print(f"(num_workers ajustado de {args.num_workers} a {effective_workers}: "
              f"este chunk solo tiene {instances_in_this_chunk} instancias)")

    result = benchmark.execute_single_n_chunk(
        args.num_nodes, chunk_index=args.chunk_index, num_chunks=args.num_chunks,
        num_workers=effective_workers,
    )

    filename = f"n_{args.num_nodes}_chunk_{args.chunk_index}_of_{args.num_chunks}.h5"
    path = benchmark.save_result_hdf5(result, args.outdir, filename=filename)

    print(f"\n✔ n={args.num_nodes} chunk {args.chunk_index + 1}/{args.num_chunks} "
          f"({len(result['instance_indices'])} instancias) "
          f"beta_parcial={result['beta_chunk']:.4f} ± {result['beta_std_chunk']:.4f}")
    print(f"✔ Resultado del chunk guardado en: {path}")
    print("  (recuerda: esto es PARCIAL — hace falta merge_chunks_qscore_PCE.py "
          "para combinar todos los chunks de este n antes de agregar)")


if __name__ == "__main__":
    main()