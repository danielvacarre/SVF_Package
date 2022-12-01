from docplex.mp.model import Model
from svf_package.efficiency.efficiency_method import EfficiencyMethod

class FDH(EfficiencyMethod):
    def __init__(self, inputs, outputs, data, methods, df_estimation=None):
        super().__init__(inputs, outputs, data, methods, df_estimation)

    def calculate_ri(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_dim_x = len(self.inputs)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_dim_y = len(self.outputs)
            # Número de observaciones del problema
            n_obs = len(Y)
            # Variable landa
            name_landa = range(0, n_obs)
            mdl = Model("DEA BCC INPUT ORIENTED")
            # Variables
            # Variable theta
            theta = mdl.continuous_var(name="theta", ub=1e33, lb=0)
            # Variable landa
            landa_var = mdl.binary_var_dict(name_landa, name="landa")
            # Función objetivo
            mdl.minimize(theta)
            # Restricciones
            for j in range(n_dim_x):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= theta * x[obs][j]
                )
            for r in range(n_dim_y):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[obs][r]
                )
            mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = round(mdl.solution["theta"], 3)
            else:
                eff = 0
            list_eff.append(eff)
        return list_eff

    def calculate_ro(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_dim_x = len(self.inputs)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_dim_y = len(self.outputs)
            # Número de observaciones del problema
            n_obs = len(Y)
            # Variable landa
            name_landa = range(0, n_obs)
            mdl = Model("DEA BCC OUTPUT ORIENTED")
            # Variables
            # Variable phi
            phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
            # Variable landa
            landa_var = mdl.binary_var_dict(name_landa, name="landa")
            # Función objetivo
            mdl.maximize(phi)
            # Restricciones
            for j in range(n_dim_x):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[obs][j]
                )
            for r in range(n_dim_y):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= phi * y[obs][r]
                )
            mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = round(mdl.solution["phi"], 3)
            else:
                eff = 0
            list_eff.append(eff)
        return list_eff

    def calculate_ddf(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_dim_x = len(self.inputs)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_dim_y = len(self.outputs)
            # Número de observaciones del problema
            n_obs = len(Y)
            # Variable landa
            name_landa = range(0, n_obs)
            mdl = Model("DEA DIRECTIONAL DISTANCE")
            # Variables
            # Variable beta
            beta = mdl.continuous_var(name="beta", ub=1e33, lb=0)
            # Variable landa
            landa_var = mdl.binary_var_dict(name_landa, name="landa")
            # Función objetivo
            mdl.maximize(beta)
            # Restricciones
            for j in range(n_dim_x):
                g = x[obs][j]
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[obs][j] - beta * g
                )
            for r in range(n_dim_y):
                g = y[obs][r]
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[obs][r] + beta * g
                )
            mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = round(mdl.solution["beta"], 3)
            else:
                eff = 0
            list_eff.append(eff)
        return list_eff

    def calculate_wa(self):
        w_inp = self.calculate_wa_w_inp()
        w_out = self.calculate_wa_w_out()
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_dim_x = len(self.inputs)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_dim_y = len(self.outputs)
            # Número de observaciones del problema
            n_obs = len(Y)
            mdl = Model("DEA WEIGHTED ADDITIVE")
            # Variables
            # Variable s
            name_s_neg = range(0, n_dim_x)
            s_neg_var = mdl.continuous_var_dict(name_s_neg, ub=1e+33, lb=0, name='s_neg')
            name_s_pos = range(0, n_dim_y)
            s_pos_var = mdl.continuous_var_dict(name_s_pos, ub=1e+33, lb=0, name='s_pos')
            # Variable landa
            name_landa = range(0, n_obs)
            landa_var = mdl.binary_var_dict(name_landa, name="landa")
            # Función objetivo
            mdl.maximize(mdl.sum(s_neg_var[j] * w_inp[j] for j in range(n_dim_x)) +
                         mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_dim_y)))
            # Restricciones
            for j in range(n_dim_x):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[obs][j] - s_neg_var[j]
                )

            for r in range(n_dim_y):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[obs][r] + s_pos_var[r]
                )

            mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)

            msol = mdl.solve()
            if msol is not None:
                eff = round(mdl.solution.get_objective_value(), 3)
            else:
                eff = 0
            # print(mdl.export_to_string())
            list_eff.append(eff)
        return list_eff

    def calculate_rui(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_dim_x = len(X.columns)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_dim_y = len(Y.columns)
            # Número de observaciones del problema
            n_obs = len(Y)
            mdl = Model("FDH INPUT ORIENTED RUSSELL")
            # Variables
            # Variable theta
            name_theta = range(0, n_dim_x)
            theta_var = mdl.continuous_var_dict(name_theta, name="theta", ub=1, lb=0)
            # Variable landa
            name_landa = range(0, n_obs)
            landa_var = mdl.binary_var_dict(name_landa, name="landa")
            # Función objetivo
            mdl.minimize(mdl.sum(theta_var[j] for j in range(n_dim_x)) / n_dim_x)
            # Restricciones
            for j in range(n_dim_x):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= theta_var[j] * x[obs][j]
                )
            for r in range(n_dim_y):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[obs][r]
                )
            mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = mdl.solution.get_objective_value()
            else:
                eff = 0
            list_eff.append(eff)
        return list_eff

    def calculate_ruo(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_dim_x = len(X.columns)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_dim_y = len(Y.columns)
            # Número de observaciones del problema
            n_obs = len(Y)
            mdl = Model("DEA OUTPUT ORIENTED RUSSELL")
            # Variables
            # Variable phi
            name_phi = range(0, n_dim_y)
            phi_var = mdl.continuous_var_dict(name_phi, name="phi", ub=1e33, lb=1)
            # Variable landa
            name_landa = range(0, n_obs)
            landa_var = mdl.binary_var_dict(name_landa, name="landa")
            # Función objetivo
            mdl.maximize(mdl.sum(phi_var[r] for r in range(n_dim_y)) / n_dim_y)
            # Restricciones
            for j in range(n_dim_x):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[obs][j]
                )
            for r in range(n_dim_y):
                mdl.add_constraint(
                    mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[obs][r] * phi_var[r]
                )
            mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = mdl.solution.get_objective_value()
            else:
                eff = 0
            # print(mdl.export_to_string())
            list_eff.append(eff)
        return list_eff

    def calculate_erg(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_dim_x = len(X.columns)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_dim_y = len(Y.columns)
            # Número de observaciones del problema
            n_obs = len(Y)

            mdl = Model("DEA ENHANCED RUSSELL GRAPH")
            # Variables

            # Variable beta
            beta_var = mdl.continuous_var(name="beta", ub=1e33, lb=0)

            # Variable t-
            name_t_neg = range(0, n_dim_x)
            t_neg_var = mdl.continuous_var_dict(name_t_neg, name="t_neg", ub=1e33, lb=0)

            # Variable t+
            name_t_pos = range(0, n_dim_y)
            t_pos_var = mdl.continuous_var_dict(name_t_pos, name="t_pos", ub=1e33, lb=0)

            # Variable landa
            name_landa = range(0, n_obs)
            landa_var = mdl.binary_var_dict(name_landa, name="landa")

            # Función objetivo
            summa = mdl.sum(t_neg_var[j]/x[obs][j] for j in range(n_dim_x))
            mdl.minimize(beta_var - (1/n_dim_x) * summa)

            # Restricciones

            # R1
            summa1 = mdl.sum(t_pos_var[r]/y[obs][r] for r in range(n_dim_y))
            mdl.add_constraint(
                beta_var + (1/n_dim_y) * summa1 == 1
            )

            # R2
            for j in range(n_dim_x):
                mdl.add_constraint(
                    -beta_var * x[obs][j] +
                    mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) +
                    t_neg_var[j]
                    == 0
                )

            # R3
            for r in range(n_dim_y):
                mdl.add_constraint(
                    -beta_var * y[obs][r] +
                    mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) -
                    t_pos_var[r]
                    == 0
                )

            # R4
            mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == beta_var)

            msol = mdl.solve()
            if msol is not None:
                eff = mdl.solution.get_objective_value()
            else:
                eff = 0
            # print(mdl.export_to_string())
            list_eff.append(eff)
        return list_eff