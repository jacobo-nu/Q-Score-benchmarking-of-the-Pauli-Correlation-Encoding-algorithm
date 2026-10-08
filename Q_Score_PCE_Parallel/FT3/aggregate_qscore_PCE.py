"""
Agregador de resultados Q-Score-PCE (flujo paralelo / job array).

Uso:
    python aggregate_qscore_PCE.py --run-id <RUN_ID>
    python aggregate_qscore_PCE.py --run-id <RUN_ID> --cleanup-logs
"""

import argparse
import glob
import json
import os
import re
import shutil
from time import strftime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_duration(seconds: float) -> str:
    """Convierte segundos a 'd:hh:mm:ss', igual que en qscore_PCE.py."""
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}:{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Agregador de resultados Q-Score-PCE")
    parser.add_argument("--run-id", type=str, default=None,
                         help="Si se indica, usa Resultados/PCE/<run-id> como --indir, "
                              "logs/<run-id> como carpeta de logs y Imagenes/PCE/<run-id> "
                              "como carpeta de figuras, sin necesidad de escribir las "
                              "rutas completas.")
    parser.add_argument("--indir", type=str, default="Resultados/PCE",
                         help="Directorio con los ficheros n_<N>.json (ignorado si se usa --run-id)")
    parser.add_argument("--logs-dir", type=str, default=None,
                         help="Directorio con los .out/.err de este run (ignorado si se usa --run-id)")
    parser.add_argument("--img-dir", type=str, default="Imagenes/PCE",
                         help="Directorio donde guardar la figura (ignorado si se usa --run-id, "
                              "o si se pasa --out con una ruta explícita)")
    parser.add_argument("--out", type=str, default=None,
                         help="Ruta de salida de la figura. Si no se indica, "
                              "se genera un nombre único con timestamp y parámetros "
                              "dentro de --img-dir.")
    parser.add_argument("--cleanup-logs", action="store_true",
                         help="Tras generar la figura, concatena todos los .out/.err de "
                              "logs-dir en un único summary.log y borra los ficheros "
                              "individuales. La información numérica ya vive en los JSON, "
                              "así que los logs crudos por tarea suelen ser prescindibles.")
    parser.add_argument("--run-ids", type=str, nargs="+", default=None,
                         help="MODO COMPARACIÓN: dos o más RUN_ID a combinar en una misma "
                              "gráfica, cada uno con su propio color. Si se indica, ignora "
                              "--run-id/--indir/--cleanup-logs y usa este modo en su lugar.")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                         help="Etiquetas de leyenda para --run-ids, en el mismo orden. "
                              "Si no se indican, se autogeneran a partir de los campos "
                              "que varíen entre runs (shots, k).")
    args = parser.parse_args()

    if args.run_id is not None:
        args.indir = os.path.join("Resultados", "PCE", args.run_id)
        args.logs_dir = os.path.join("logs", args.run_id)
        args.img_dir = os.path.join("Imagenes", "PCE", args.run_id)

    return args


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_hdf5(path: str) -> dict:
    """Lee un n_<N>.h5 y lo devuelve como el mismo dict que producía el JSON."""
    import h5py

    with h5py.File(path, "r") as f:
        result = {}
        for key, value in f.attrs.items():
            # Los atributos llegan como tipos numpy (np.int64, np.bool_, ...);
            # se convierten a tipos Python nativos para que el resto del
            # agregador (formateos, pandas) los trate igual que con JSON.
            result[key] = value.item() if hasattr(value, "item") else value
        result["cut_sizes"] = f["cut_sizes"][()].tolist()
    return result


def load_results(indir: str):
    """
    Carga los resultados por n de un run, aceptando tanto el formato nuevo
    (HDF5, n_<N>.h5) como el antiguo (JSON, n_<N>.json). Si por lo que sea
    existieran ambos para un mismo n, gana el HDF5 (formato más reciente).
    """
    json_paths = glob.glob(os.path.join(indir, "n_*.json"))
    h5_paths = glob.glob(os.path.join(indir, "n_*.h5"))
    if not json_paths and not h5_paths:
        raise FileNotFoundError(f"No se encontraron ficheros n_*.json ni n_*.h5 en {indir}")

    by_n = {}
    for path in json_paths:
        r = _load_json(path)
        by_n[r["num_nodes"]] = r
    for path in h5_paths:
        r = _load_hdf5(path)
        by_n[r["num_nodes"]] = r  # sobreescribe el JSON si coexisten

    results = sorted(by_n.values(), key=lambda r: r["num_nodes"])
    return results


def _natural_key(path):
    """
    Extrae el número de tarea al final del nombre (antes de la extensión)
    para ordenar 0,1,2...10 en vez de alfabético (0,1,10,2...). Sirve tanto
    para el patrón de array (qscore_pce_<jobid>_<idx>.out) como para el de
    packed (qscore_pce_n<N>.out) — en ambos casos el número relevante es
    el que aparece justo al final del nombre de fichero.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", stem)
    return (int(match.group(1)) if match else float("inf"), stem)


def cleanup_logs(logs_dir):
    if logs_dir is None:
        return
    out_files = sorted(glob.glob(os.path.join(logs_dir, "*.out")), key=_natural_key)
    err_files = sorted(glob.glob(os.path.join(logs_dir, "*.err")), key=_natural_key)
    log_files = out_files + err_files
    if not log_files:
        return

    summary_path = os.path.join(logs_dir, "summary.log")
    with open(summary_path, "w", encoding="utf-8") as summary:
        # Todos los .out primero (en un bloque único), luego todos los .err.
        # Antes se intercalaban por orden alfabético (_0.err, _0.out, _1.err...),
        # lo que dejaba la salida útil (.out) repartida por el medio del
        # fichero en vez de junta al principio.
        summary.write(f"\n{'#'*70}\n### STDOUT — {len(out_files)} ficheros .out\n{'#'*70}\n")
        for path in out_files:
            summary.write(f"\n{'='*70}\n{os.path.basename(path)}\n{'='*70}\n")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                summary.write(f.read())

        summary.write(f"\n{'#'*70}\n### STDERR — {len(err_files)} ficheros .err\n{'#'*70}\n")
        for path in err_files:
            summary.write(f"\n{'='*70}\n{os.path.basename(path)}\n{'='*70}\n")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                summary.write(f.read())

    for path in log_files:
        os.remove(path)

    print(f"✔ {len(log_files)} logs individuales concatenados en '{summary_path}' y eliminados.")


def _auto_labels(all_results):
    """
    Genera etiquetas automáticas para el modo comparación a partir de los
    campos que realmente varíen entre runs (shots, k, maxiter). Si todos
    los runs comparten el mismo valor de un campo, no se incluye en la
    etiqueta (sería ruido repetido); si un campo varía, se muestra en
    todas las etiquetas para que la leyenda sea autoexplicativa por sí sola.
    """
    shots_vals = {r[0].get("shots", 0) for _, r in all_results}
    k_vals = {r[0]["k"] for _, r in all_results}
    maxiter_vals = {r[0].get("pce_maxiter") for _, r in all_results}

    labels = []
    for run_id, results in all_results:
        r0 = results[0]
        parts = []
        if len(shots_vals) > 1:
            s = r0.get("shots", 0)
            parts.append(f"shots={s}" if s else "exact")
        if len(k_vals) > 1:
            parts.append(f"k={r0['k']}")
        if len(maxiter_vals) > 1 and r0.get("pce_maxiter") is not None:
            parts.append(f"maxiter={r0['pce_maxiter']}")
        labels.append(", ".join(parts) if parts else run_id)
    return labels


def main_compare(args):
    """
    Modo comparación: combina varios runs (--run-ids) en una sola gráfica,
    un color por run, para comparar p.ej. mismo número de shots con
    distinta compresión k, o al revés. No dibuja las líneas verticales de
    salto de qubits (con varios runs a la vez se solaparían y ensuciarían
    la gráfica) y no genera CSV combinado — cada run conserva su propio
    CSV individual, generado normalmente con --run-id.
    """
    run_ids = args.run_ids
    labels = args.labels
    if labels is not None and len(labels) != len(run_ids):
        raise SystemExit(
            f"--labels tiene {len(labels)} elementos pero --run-ids tiene "
            f"{len(run_ids)}; deben coincidir uno a uno."
        )

    all_results = []
    for run_id in run_ids:
        indir = os.path.join("Resultados", "PCE", run_id)
        results = load_results(indir)
        all_results.append((run_id, results))

    if labels is None:
        labels = _auto_labels(all_results)

    fig = plt.figure()
    ax = plt.axes()
    plt.axhline(0.2, color="red", linestyle="dashed", label="Threshold")

    cmap = plt.get_cmap("tab10")
    all_nodes = set()
    combined_rows = []
    for i, ((run_id, results), label) in enumerate(zip(all_results, labels)):
        nodes_list = [r["num_nodes"] for r in results]
        beta_list = [r["beta"] for r in results]
        beta_std_list = [r["beta_std"] for r in results]
        all_nodes.update(nodes_list)
        color = cmap(i % 10)
        plt.errorbar(nodes_list, beta_list, yerr=beta_std_list, fmt="-o",
                     capsize=6, markersize=6, color=color, label=label)

        qscore = 0
        for r in results:
            if r["success"]:
                qscore = r["num_nodes"]
            row = dict(r)
            row["run_id"] = run_id
            row["label"] = label
            combined_rows.append(row)
        print(f"[{label}] (run_id={run_id})  Q-Score (PCE) final: {qscore}")

    ax.set_ylabel(r"Q-score ratio $\beta(n)$")
    ax.set_xlabel("Number of nodes $(n)$")
    plt.xticks(sorted(all_nodes), rotation=90)
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True)
    plt.title(f"Q-score (PCE) —  {len(run_ids)} different maxiters", fontsize=9)
    plt.gcf().set_dpi(250)

    run_timestamp = strftime("%Y%m%d-%H%M%S")
    if args.out is not None:
        out_path = args.out
    else:
        img_dir = args.img_dir
        fig_name = f"qscore_pce_compare_{len(run_ids)}runs_{run_timestamp}.png"
        os.makedirs(img_dir, exist_ok=True)
        out_path = os.path.join(img_dir, fig_name)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"\n✔ Figura comparativa guardada en: {out_path}")

    # CSV combinado, con columna run_id/label para poder filtrar luego.
    df = pd.DataFrame(combined_rows)
    ordered_cols = [c for c in [
        "run_id", "label", "num_nodes", "num_qubits", "num_instances", "beta",
        "beta_std", "success", "duration_seconds", "k", "seed", "pce_optimizer",
        "pce_maxiter", "simulator", "device", "device_actual", "shots",
        "timestamp", "cut_sizes",
    ] if c in df.columns]
    df = df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"✔ Tabla combinada (CSV) guardada en: {csv_path}")


def main():
    args = parse_args()

    if args.run_ids is not None:
        main_compare(args)
        return

    results = load_results(args.indir)

    nodes_list = [r["num_nodes"] for r in results]
    beta_list = [r["beta"] for r in results]
    beta_std_list = [r["beta_std"] for r in results]
    num_instances = results[0]["num_instances"]

    # --- Parámetros para el título: k, optimizer, maxiter, seed ---
    # Se toman del primer resultado; si alguna tarea usó valores distintos
    # (por ejemplo, si se relanzó solo un n con otra config), se avisa.
    k = results[0]["k"]
    optimizer = results[0]["pce_optimizer"]
    maxiter = results[0]["pce_maxiter"]
    seed = results[0]["seed"]
    simulator = results[0].get("simulator")
    device = results[0].get("device")
    backend = results[0].get("backend")
    shots = results[0].get("shots", 0)  # retrocompatible: runs sin este campo = modo exacto

    inconsistent = [
        r["num_nodes"] for r in results
        if (r["k"], r["pce_optimizer"], r["pce_maxiter"], r["seed"]) != (k, optimizer, maxiter, seed)
    ]
    if inconsistent:
        print(f"⚠ Aviso: los tamaños {inconsistent} usaron parámetros distintos "
              f"al resto (k/optimizer/maxiter/seed). El título muestra los del "
              f"primer resultado (n={results[0]['num_nodes']}).")

    qscore = 0
    for r in results:
        if r["success"]:
            qscore = r["num_nodes"]
        status = "PASSED" if r["success"] else "FAILED"
        qubits_str = f"  qubits={r['num_qubits']}" if r.get("num_qubits") is not None else ""
        duration_str = f"  t={format_duration(r['duration_seconds'])}" if r.get("duration_seconds") is not None else ""
        r_shots = r.get("shots", 0)
        shots_str = f"  shots={r_shots}" if r_shots else "  exact"
        print(f"n={r['num_nodes']:3d}  beta={r['beta']:.4f} ± {r['beta_std']:.4f}  ({status}){qubits_str}{duration_str}{shots_str}")

    print(f"\nQ-Score (PCE) final: {qscore}")

    mismatched = [
        r["num_nodes"] for r in results
        if r.get("device") is not None and r.get("device_actual") is not None
        and r["device"] != r["device_actual"]
    ]
    if mismatched:
        print(f"⚠ Aviso: en n={mismatched}, Aer usó un device distinto al solicitado "
              f"(revisa 'device' vs 'device_actual' — posible fallback silencioso a CPU).")

    fig = plt.figure()
    ax = plt.axes()
    plt.axhline(0.2, color="red", linestyle="dashed", label="Threshold")
    plt.errorbar(nodes_list, beta_list, yerr=beta_std_list, fmt="-o",
                 capsize=10, markersize=8, color="#2E8B57", label="PCE approximation ratio")

    # --- Líneas verticales en los saltos de número de qubits ---
    # Solo se marca el primer n en el que se empieza a usar un qubit más
    # (no se repite la línea en cada n posterior que siga usando ese mismo número).
    qubits_list = [r.get("num_qubits") for r in results]
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
    else:
        print("⚠ Aviso: no todos los resultados tienen 'num_qubits' guardado "
              "(JSON de una versión anterior) — no se dibujan las líneas de qubits.")

    ax.set_ylabel(r"Q-score ratio $\beta(n)$")
    ax.set_xlabel("Number of nodes $(n)$")
    plt.xticks(nodes_list, rotation=90)
    plt.legend(loc="lower left")
    plt.grid(True)
    title = (
        f"Q-score (PCE), {num_instances} instances\n"
        f"k={k}, optimizer={optimizer}, maxiter={maxiter}, seed={seed}"
    )
    # Igual que en el flujo serie: en modo shots el método interno de Aer
    # (statevector/mps/...) no es lo relevante para el título — se muestra
    # el simulador SOLO si es modo exacto, y los shots SOLO si shots>0,
    # nunca los dos a la vez.
    # En hardware real (backend == "QMIO_REAL") 'device' sigue valiendo
    # "CPU" (viene de PCE_DEVICE, que nunca se apunta a la QPU), así que
    # el device mostrado se fuerza a "QPU" — no tiene sentido decir
    # CPU/GPU cuando en realidad se ejecutó en el chip.
    device_label = "QPU" if backend == "QMIO_REAL" else device
    if shots:
        title += f"\nshots={shots}" + (f" ({device_label})" if device_label is not None else "")
    elif simulator is not None:
        title += f"\n{simulator}" + (f" ({device_label})" if device_label is not None else "")
    plt.title(title, fontsize=9)
    plt.gcf().set_dpi(250)

    # --- Nombre de fichero único: evita sobreescribir figuras entre runs ---
    run_timestamp = strftime("%Y%m%d-%H%M%S")
    if args.out is not None:
        out_path = args.out
    else:
        fig_name = (
            f"qscore_pce_n{min(nodes_list)}-{max(nodes_list)}"
            f"_k{k}_{optimizer}_maxiter{maxiter}_seed{seed}"
            f"_{run_timestamp}.png"
        )
        os.makedirs(args.img_dir, exist_ok=True)
        out_path = os.path.join(args.img_dir, fig_name)

    # bbox_inches='tight' recalcula los márgenes según el contenido real,
    # para que las etiquetas rotadas del eje x no corten el nombre del eje.
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\n✔ Figura guardada en: {out_path}")

    # --- Exportar los resultados agregados a pandas (CSV) ---
    # Una fila por n, con todos los metadatos escalares. cut_sizes se
    # incluye como lista (pandas la serializa como texto en el CSV); los
    # datos crudos originales siguen en los .h5/.json de Resultados/.
    df = pd.DataFrame(results)
    ordered_cols = [c for c in [
        "num_nodes", "num_qubits", "num_instances", "beta", "beta_std",
        "success", "duration_seconds", "k", "seed", "pce_optimizer",
        "pce_maxiter", "simulator", "device", "device_actual", "shots", "timestamp", "cut_sizes",
    ] if c in df.columns]
    df = df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]

    csv_name = (
        f"qscore_pce_n{min(nodes_list)}-{max(nodes_list)}"
        f"_k{k}_{optimizer}_maxiter{maxiter}_seed{seed}"
        f"_{run_timestamp}.csv"
    )
    csv_path = os.path.join(args.indir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"✔ Tabla pandas (CSV) guardada en: {csv_path}")

    if args.cleanup_logs:
        cleanup_logs(args.logs_dir)


if __name__ == "__main__":
    main()