from os import listdir
from os.path import isfile, join
from pandas import read_csv, DataFrame, unique
from svf_package.cv.cv import CrossValidation
from svf_package.simulation import Simulation
from svf_package.svf_functions import create_SVF

def calculate_d(n_obs):
    d = list()
    for i in range(1, 11):
        n = int(round(0.1 * i * n_obs, 0))
        if n > 0:
            d.append(n)
    d = unique(d).tolist()
    return d

if __name__ == '__main__':

    path_datos = "./data/data_1_30/"

    C = [0.0001, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100]
    eps = [0, 0.001, 0.01, 0.1, 1, 10, 15, 25, 30, 50, 75, 100]
    n_folds = 5

    # DATOS DEL ESCENARIO
    n_dim_x = 1
    n_dim_y = 1
    n_dmus = 30

    D = calculate_d(n_dmus)

    inputs = []
    for i in range(0, n_dim_x):
        col = "x"+str(i+1)
        inputs.append(col)
    outputs = []
    if n_dim_y != 1:
      for i in range(0, n_dim_x):
        col = "y"+str(i+1)
        outputs.append(col)
    else:
      outputs = ['y']

    method = "SSVF"

    simulation = Simulation()

    files = [f for f in listdir(path_datos) if isfile(join(path_datos, f))]
    for f in files:
        if f != "desktop.ini":

            file_load = path_datos + f
            nombre = f.split('.')[0]
            split = f.split("_")
            simulation.nombre = nombre
            simulation.scenario = split[1]
            simulation.size = nombre.split("_")[2]

            data_simulation = read_csv(file_load, sep=";")

            simulation.cv = CrossValidation(method, inputs, outputs, data_simulation, C, eps, D, verbose=True,
                                               n_folds=n_folds)
            simulation.cv.cv()

            simulation.guardar_registro_fichero(simulation.cv.results, "CV")

            simulation.svf_obj = create_SVF(method, inputs, outputs, data_simulation,
                                 simulation.cv.best_C,
                                 simulation.cv.best_eps,
                                 simulation.cv.best_d)
            simulation.svf_obj.train()
            simulation.svf_obj.solve()

            df = simulation.svf_obj.get_df_all_estimation()
            df["y_fron"] = simulation.true_fontier(simulation.scenario, data_simulation, inputs)

            simulation.guardar_registro_fichero(df, "EST")

            simulation.hiper_list.append(simulation.get_hip_metrics(f, simulation.svf_obj.C, simulation.svf_obj.eps, simulation.svf_obj.d))
            simulation.bias_list.append(simulation.get_bias_metrics(f, df))
            simulation.mse_list.append(simulation.get_mse_metrics(f, df))
            simulation.time_list.append(simulation.get_time_metrics(f))

            simulation.hiper_df = DataFrame(simulation.hiper_list)
            simulation.mse_df = DataFrame(simulation.mse_list)
            simulation.bias_df = DataFrame(simulation.bias_list)
            simulation.time_df = DataFrame(simulation.time_list)

            simulation.guardar_registro_fichero(simulation.hiper_df,"HIP")
            simulation.guardar_registro_fichero(simulation.mse_df,"MSE")
            simulation.guardar_registro_fichero(simulation.bias_df,"BIAS")
            simulation.guardar_registro_fichero(simulation.time_df,"TIME")

    simulation.get_agg_hip_metrics()
    simulation.get_agg_mse_metrics()
    simulation.get_agg_bias_metrics()
    simulation.get_agg_time_metrics()

    simulation.guardar_resultados_fichero(simulation.hiper_df_agg, "HIP")
    simulation.guardar_resultados_fichero(simulation.mse_df_agg, "MSE")
    simulation.guardar_resultados_fichero(simulation.bias_df_agg, "BIAS")
    simulation.guardar_resultados_fichero(simulation.time_df_agg, "TIME")
