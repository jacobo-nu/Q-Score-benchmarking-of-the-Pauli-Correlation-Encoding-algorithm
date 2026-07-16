

Optimiz = ["DIFFERENTIALEVOLUTION"]
k = [2]

# Número de iteraciones a llevar a cabo
maxiter = 10

# Número de shots a llevar a cabo
n_shots = 1 # En este caso no se van a usar porque se va a tener simulaciónn exacta
optimizer_params = None
nqpus = None
cunqa_str = "Simulation"
family_name = None

G

alpha = 1.5 * num_qubits(num_ver, k)
beta = 0.5

 # === 4. Ejecutar MaxCut ===
dic_resultado, subcarpeta, ruta_csv, ruta_csv_iter = ejecutar_maxcut(
    G=G,
    optimizer=optimizer,
    optimizer_params=opt_params,
    num_ver=num_ver,
    k=k,
    alpha=alpha,
    beta=beta,
    maxiter=maxiter,
    n_shots=n_shots,
    nqpus=nqpus,
    cunqa_str_arg=cunqa_str,
    family_name=family_name
)