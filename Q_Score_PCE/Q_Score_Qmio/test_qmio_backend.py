"""
test_qmio_backend.py
=====================
Prueba mínima e independiente del pipeline PCE, para verificar que la
integración con QmioBackend (hardware real) y FakeQmio (emulador con
ruido) funciona en tu entorno ANTES de lanzar nada del pipeline completo
contra ellos.

NO VERIFICADO POR mí en ejecución real — basado en la documentación
oficial de qmiotools 0.2.1:
  https://gomeztato.github.io/qmiotools/integrations.qiskitqmio.html
  https://cesga-docs.gitlab.io/qmio-user-guide/interfaces.html
  https://cesga-docs.gitlab.io/qmio-user-guide/examples.html

Uso:
    # FakeQmio (no necesita partición qpu, puedes probarlo en 'ilk'):
    python test_qmio_backend.py --backend fake

    # Hardware real (SOLO desde la partición 'qpu', ver job_qpu.sh):
    python test_qmio_backend.py --backend real
"""

import argparse

from qiskit import QuantumCircuit, transpile


def parse_args():
    parser = argparse.ArgumentParser(description="Prueba mínima de QmioBackend/FakeQmio")
    parser.add_argument("--backend", choices=["real", "fake"], required=True,
                         help="'real' = hardware QMIO (requiere partición qpu); "
                              "'fake' = emulador con ruido calibrado (cualquier nodo)")
    parser.add_argument("--shots", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n=== Prueba mínima QmioBackend/FakeQmio (backend={args.backend}) ===\n")

    # Circuito de Bell — mismo circuito de ejemplo que usa la doc de CESGA
    # para el hardware real, así que si algo falla aquí, no es por el
    # circuito en sí sino por la integración backend/entorno.
    c = QuantumCircuit(2)
    c.h(0)
    c.cx(0, 1)
    c.measure_all()

    if args.backend == "real":
        from qmiotools.integrations.qiskitqmio import QmioBackend
        print("Conectando a QmioBackend() — esto imprime el fichero de calibración usado...")
        backend = QmioBackend()
    else:
        from qmiotools.integrations.qiskitqmio import FakeQmio
        print("Creando FakeQmio() con la última calibración...")
        backend = FakeQmio()

    print(f"Backend creado: {backend!r}")

    print("Transpilando (optimization_level=2)...")
    c_t = transpile(c, backend, optimization_level=2)

    print(f"Ejecutando con shots={args.shots}...")
    job = backend.run(c_t, shots=args.shots)
    result = job.result()

    # Punto a verificar: si esto falla con TypeError/AttributeError,
    # prueba result.get_counts() SIN argumento en vez de con el circuito
    # — no está confirmado cuál de las dos formas soporta QmioJob.
    try:
        counts = result.get_counts(c_t)
    except Exception as e:
        print(f"⚠ get_counts(c_t) falló ({e}), probando get_counts() sin argumento...")
        counts = result.get_counts()

    print(f"\nCounts: {counts}")

    # Para un estado de Bell (|00> + |11>)/sqrt(2), se espera ~50% '00'
    # y ~50% '11' (con algo de ruido si backend='fake', o ruido real si
    # backend='real'). Si sale otra cosa muy distinta, algo no va bien
    # (medida en el orden de bits equivocado, circuito mal transpilado...).
    print("\nSi ves aprox. la mitad en '00' y la mitad en '11' (con algo de ruido "
          "si es fake/real), la integración funciona correctamente.")

    if hasattr(backend, "disconnect"):
        backend.disconnect()


if __name__ == "__main__":
    main()