"""
merge_chunks_qscore_PCE.py
============================
Fusiona los ficheros de chunk (n_<N>_chunk_<i>_of_<C>.h5, generados por
run_qscore_PCE_chunk.py) de un mismo RUN_ID en el n_<N>.h5 final que espera
aggregate_qscore_PCE.py — reconstruye el cut_sizes completo (en el orden
correcto de instancia) y recalcula beta/beta_std sobre el TOTAL de
instancias, no sobre cada chunk por separado (los beta_chunk/beta_std_chunk
de cada fichero de chunk son solo informativos).

Es el segundo nivel de agregación: primero este script combina chunks -> un
n_<N>.h5 por tamaño de grafo, y DESPUÉS aggregate_qscore_PCE.py (sin ningún
cambio) combina esos n_<N>.h5 entre sí como siempre.

Uso:
    python merge_chunks_qscore_PCE.py --run-id <RUN_ID>
    python merge_chunks_qscore_PCE.py --chunks-dir Resultados/PCE/<RUN_ID>/chunks \
        --outdir Resultados/PCE/<RUN_ID>
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import h5py
import numpy as np

from qscore_PCE import compute_beta, is_successful, format_duration, qcvv_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Fusiona chunks de Q-Score-PCE en n_<N>.h5 finales")
    parser.add_argument("--run-id", type=str, default=None,
                         help="Si se indica, usa Resultados/PCE/<run-id>/chunks como --chunks-dir "
                              "y Resultados/PCE/<run-id> como --outdir.")
    parser.add_argument("--chunks-dir", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--keep-chunks", action="store_true",
                         help="No borrar los ficheros de chunk individuales tras fusionar "
                              "(por defecto SÍ se borran, ya que su información ya vive "
                              "íntegra en el n_<N>.h5 final).")
    args = parser.parse_args()

    if args.run_id is not None:
        args.chunks_dir = os.path.join("Resultados", "PCE", args.run_id, "chunks")
        args.outdir = os.path.join("Resultados", "PCE", args.run_id)

    if args.chunks_dir is None or args.outdir is None:
        parser.error("Indica --run-id, o --chunks-dir y --outdir explícitos.")

    return args


def _load_chunk(path: str) -> dict:
    with h5py.File(path, "r") as f:
        result = {}
        for key, value in f.attrs.items():
            if hasattr(value, "size") and getattr(value, "size", 1) > 1:
                # Atributo multi-elemento (p.ej. instance_indices) — .item()
                # solo funciona con arrays de un único elemento, así que aquí
                # se convierte a lista en vez de intentar un escalar.
                result[key] = value.tolist()
            elif hasattr(value, "item"):
                result[key] = value.item()
            else:
                result[key] = value
        result["cut_sizes"] = f["cut_sizes"][()].tolist()
        if "instance_indices" in result:
            result["instance_indices"] = [int(i) for i in np.atleast_1d(result["instance_indices"])]
    return result


def find_chunk_groups(chunks_dir: str):
    """
    Agrupa los ficheros de chunk por num_nodes. Devuelve
    {num_nodes: [dicts de chunk, ordenados por chunk_index]}.
    """
    pattern = os.path.join(chunks_dir, "n_*_chunk_*_of_*.h5")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No se encontraron ficheros de chunk en {chunks_dir} (patrón: {pattern})")

    groups = defaultdict(list)
    for path in paths:
        chunk = _load_chunk(path)
        chunk["_path"] = path
        groups[chunk["num_nodes"]].append(chunk)

    for num_nodes in groups:
        groups[num_nodes].sort(key=lambda c: c["chunk_index"])

    return groups


def merge_one_n(num_nodes: int, chunks: list) -> dict:
    """
    Combina los chunks de un mismo num_nodes en un único resultado,
    verificando que la cobertura de instancias es completa (sin huecos
    ni duplicados) antes de aceptar el resultado como válido.
    """
    num_chunks_expected = chunks[0]["num_chunks"]
    num_instances = chunks[0]["num_instances"]

    if len(chunks) != num_chunks_expected:
        raise ValueError(
            f"n={num_nodes}: se esperaban {num_chunks_expected} chunks, pero solo "
            f"se encontraron {len(chunks)} — probablemente aún falta alguno por terminar "
            f"(o falló). No se fusiona hasta tener todos."
        )

    cut_sizes_by_idx = {}
    for chunk in chunks:
        if chunk["num_nodes"] != num_nodes or chunk["num_chunks"] != num_chunks_expected \
           or chunk["num_instances"] != num_instances:
            raise ValueError(
                f"n={num_nodes}: metadatos inconsistentes entre chunks de este n "
                f"(num_nodes/num_chunks/num_instances no coinciden) — revisa {chunk['_path']}"
            )
        for idx, cut_size in zip(chunk["instance_indices"], chunk["cut_sizes"]):
            if idx in cut_sizes_by_idx:
                raise ValueError(
                    f"n={num_nodes}: instancia {idx} aparece en más de un chunk "
                    f"(revisa {chunk['_path']}) — reparto de instancias corrupto."
                )
            cut_sizes_by_idx[idx] = cut_size

    missing = sorted(set(range(num_instances)) - set(cut_sizes_by_idx))
    if missing:
        raise ValueError(f"n={num_nodes}: faltan instancias {missing} tras fusionar todos los chunks disponibles.")

    cut_sizes_list = [cut_sizes_by_idx[i] for i in range(num_instances)]

    approximation_ratio, beta_std = compute_beta(cut_sizes_list, num_nodes)
    success = is_successful(approximation_ratio)

    # duration_seconds: los chunks corrieron en PARALELO (si todo fue
    # bien), así que sumar sus duraciones no representa el tiempo de
    # pared real — se usa el máximo (aproximación del tiempo de pared,
    # asumiendo que arrancaron aprox. a la vez) y se guarda además la
    # suma como total_cpu_seconds, para no perder el dato de coste total.
    durations = [c.get("duration_seconds", 0.0) for c in chunks]
    wall_duration = max(durations) if durations else 0.0
    total_cpu_seconds = sum(durations)

    first = chunks[0]
    result = {
        "num_nodes": num_nodes,
        "num_qubits": first.get("num_qubits"),
        "num_instances": num_instances,
        "cut_sizes": cut_sizes_list,
        "beta": approximation_ratio,
        "beta_std": beta_std,
        "success": success,
        "timestamp": first.get("timestamp"),
        "duration_seconds": wall_duration,
        "total_cpu_seconds": total_cpu_seconds,
        "num_chunks_merged": num_chunks_expected,
        "k": first.get("k"),
        "seed": first.get("seed"),
        "pce_optimizer": first.get("pce_optimizer"),
        "pce_maxiter": first.get("pce_maxiter"),
        "simulator": first.get("simulator"),
        "device": first.get("device"),
        "max_parallel_threads": first.get("max_parallel_threads"),
        "shots": first.get("shots"),
        "backend": first.get("backend"),
    }

    status = "PASSED" if success else "FAILED"
    qcvv_logger.info(
        f"[PCE][merge] n={num_nodes} FUSIONADO ({num_chunks_expected} chunks, "
        f"{num_instances} instancias) beta={approximation_ratio:.4f} ± {beta_std:.4f} "
        f"({status}) tiempo_pared~={format_duration(wall_duration)} "
        f"cpu_total={format_duration(total_cpu_seconds)}"
    )

    return result


def main():
    args = parse_args()
    groups = find_chunk_groups(args.chunks_dir)

    os.makedirs(args.outdir, exist_ok=True)

    from qscore_PCE import QScoreBenchmarkPCE

    merged_paths = []
    skipped = []
    for num_nodes in sorted(groups):
        chunks = groups[num_nodes]
        try:
            result = merge_one_n(num_nodes, chunks)
        except ValueError as e:
            print(f"⚠ n={num_nodes}: {e}")
            skipped.append(num_nodes)
            continue

        path = QScoreBenchmarkPCE.save_result_hdf5(result, args.outdir)
        merged_paths.append(path)
        print(f"✔ n={num_nodes}: {len(chunks)} chunks fusionados -> {path}")

        if not args.keep_chunks:
            for chunk in chunks:
                os.remove(chunk["_path"])

    print(f"\n{len(merged_paths)} tamaños de n fusionados correctamente.")
    if skipped:
        print(f"⚠ {len(skipped)} tamaños de n SIN fusionar (incompletos): {skipped}")
        print("  Vuelve a ejecutar este script cuando terminen los chunks que faltan.")


if __name__ == "__main__":
    main()