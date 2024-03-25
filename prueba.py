from random import randint
from SVF_Package.cv.cv import CrossValidation
from SVF_Package.svf_functions import create_SVF
from pandas import read_csv, DataFrame
from docplex.mp.model import Model



if __name__ == "__main__":
    inputs = ["x1","x2"]
    outputs = ["y"]
    method = "SSVF"

    C = [1]
    eps = [0]
    d = [2]

    data_simulation = read_csv("datos.csv", sep=";")

    cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, verbose=True,n_folds=2)
    cross_validation.cv()

    svf_obj = create_SVF(method, inputs, outputs, data_simulation, cross_validation.best_C, cross_validation.best_eps, cross_validation.best_d)
    svf_obj.train()
    svf_obj.solve()

    svf_obj.get_df_estimation()
    svf_obj.get_virtual_grid_estimation()

    data = svf_obj.grid.virtual_grid.copy()

    w_i_list = list()
    w_list = list()
    for _ in range(len(outputs)):
        for _ in range(len(data)):
            valor_aleatorio = randint(1, 100)  # Generar un número aleatorio entre 1 y 100
            w_i_list.append(valor_aleatorio)
            w_list.append(w_i_list)
        # list_eff = list()
        # for obs in range(len(data)):
        #     # Datos de las variables distintas de Y
        #     X = data.filter(inputs)
        #     x = X.values.tolist()
        #     # Número de dimensiones X del problema
        #     n_dim_x = len(inputs)
        #     # Datos de las variables Y
        #     Y = data.filter(outputs)
        #     y = Y.values.tolist()
        #     # Número de dimensiones y del problema
        #     n_dim_y = len(outputs)
        #     # Número de observaciones del problema
        #     n_obs = len(Y)
        #     mdl = Model("DEA WEIGHTED ADDITIVE")
        #     # Variables
        #     # Variable s
        #     name_x_var = range(0, n_dim_y)
        #     x_var = mdl.continuous_var_dict(name_x_var, ub=1e+33, lb=0, name='x')
        #     # Variable landa
        #     name_landa = range(0, n_obs)
        #     landa_var = mdl.binary_var_dict(name_landa, name="landa")
        #     # Función objetivo
        #     mdl.maximize(mdl.sum(s_neg_var[j] * w_list[j] for j in range(n_dim_x)) +
        #                  mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_dim_y)))
        #     # Restricciones
        #     for j in range(n_dim_x):
        #         mdl.add_constraint(
        #             mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[obs][j] - s_neg_var[j]
        #         )
        #
        #     for r in range(n_dim_y):
        #         mdl.add_constraint(
        #             mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[obs][r] + s_pos_var[r]
        #         )
        #
        #     mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
        #
        #     msol = mdl.solve()
        #     if msol is not None:
        #         eff = round(mdl.solution.get_objective_value(), 3)
        #     else:
        #         eff = 0
        #     # print(mdl.export_to_string())
        #     list_eff.append(eff)
        # return list_eff