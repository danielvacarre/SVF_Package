from docplex.mp.model import Model

class DEA:
    def __init__(self, inputs, outputs, data, methods):
        self.data = data
        self.methods = methods
        self.outputs = outputs
        self.inputs = inputs
        self.model = None
        self.solution = None
        self.df_eff = None

    def calculate_dea_ri(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_inp = len(self.inputs)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_out = len(self.outputs)
            # Número de observaciones del problema
            n_obs = len(Y)
            # Variable landa
            name_landa = range(0, n_obs)
            mdl = Model("DEA BCC INPUT ORIENTED")
            # Variables
            # Variable theta
            theta = mdl.continuous_var(name="theta", ub=1e33, lb=0)
            # Variable landa
            landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
            # Función objetivo
            mdl.minimize(theta)
            # Restricciones
            for inp in range(n_inp):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * x[obs][inp] for obs in range(n_obs)) <= theta * x[obs][inp]
                )
            for r in range(n_out):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * y[obs][r] for obs in range(n_obs)) >= y[obs][r]
                )
            mdl.add_constraint(mdl.sum(landa_var[obs] for obs in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = round(mdl.solution["theta"], 3)
            else:
                eff = 0
            list_eff.append(eff)
        return list_eff

    def calculate_dea_ro(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_inp = len(self.inputs)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_out = len(self.outputs)
            # Número de observaciones del problema
            n_obs = len(Y)
            # Variable landa
            name_landa = range(0, n_obs)
            mdl = Model("DEA BCC OUTPUT ORIENTED")
            # Variables
            # Variable phi
            phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
            # Variable landa
            landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
            # Función objetivo
            mdl.maximize(phi)
            # Restricciones
            for inp in range(n_inp):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * x[obs][inp] for obs in range(n_obs)) <= x[obs][inp]
                )
            for r in range(n_out):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * y[obs][r] for obs in range(n_obs)) >= phi * y[obs][r]
                )
            mdl.add_constraint(mdl.sum(landa_var[obs] for obs in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = round(mdl.solution["phi"], 3)
            else:
                eff = 0
            list_eff.append(eff)
        return list_eff

    def calculate_dea_ddf(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_inp = len(self.inputs)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_out = len(self.outputs)
            # Número de observaciones del problema
            n_obs = len(Y)
            # Variable landa
            name_landa = range(0, n_obs)
            mdl = Model("DEA DIRECTIONAL DISTANCE")
            # Variables
            # Variable beta
            beta = mdl.continuous_var(name="beta", ub=1e33, lb=0)
            # Variable landa
            landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
            # Función objetivo
            mdl.maximize(beta)
            # Restricciones
            for inp in range(n_inp):
                g = x[obs][inp]
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * x[obs][inp] for obs in range(n_obs)) <= x[obs][inp] - beta * g
                )
            for r in range(n_out):
                g = y[obs][r]
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * y[obs][r] for obs in range(n_obs)) >= y[obs][r] + beta * g
                )
            mdl.add_constraint(mdl.sum(landa_var[obs] for obs in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = round(mdl.solution["beta"], 3)
            else:
                eff = 0
            list_eff.append(eff)
        return list_eff

    def calculate_dea_wa(self):
        w_inp = self.calculate_wa_w_inp()
        w_out = self.calculate_wa_w_out()
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_inp = len(self.inputs)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_out = len(self.outputs)
            # Número de observaciones del problema
            n_obs = len(Y)
            mdl = Model("DEA WEIGHTED ADDITIVE")
            # Variables
            # Variable s
            name_s_neg = range(0, n_inp)
            s_neg_var = mdl.continuous_var_dict(name_s_neg, ub=1e+33, lb=0, name='s_neg')
            name_s_pos = range(0, n_out)
            s_pos_var = mdl.continuous_var_dict(name_s_pos, ub=1e+33, lb=0, name='s_pos')
            # Variable landa
            name_landa = range(0, n_obs)
            landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
            # Función objetivo
            mdl.maximize(mdl.sum(s_neg_var[inp] * w_inp[inp] for inp in range(n_inp)) +
                         mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_out)))
            # Restricciones
            for inp in range(n_inp):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * x[obs][inp] for obs in range(n_obs)) <= x[obs][inp] - s_neg_var[inp]
                )

            for r in range(n_out):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * y[obs][r] for obs in range(n_obs)) >= y[obs][r] + s_pos_var[r]
                )

            mdl.add_constraint(mdl.sum(landa_var[obs] for obs in range(n_obs)) == 1)

            msol = mdl.solve()
            if msol is not None:
                eff = round(mdl.solution.get_objective_value(), 3)
            else:
                eff = 0
            # print(mdl.export_to_string())
            list_eff.append(eff)
        return list_eff

    def calculate_dea_rui(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_inp = len(X.columns)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_out = len(Y.columns)
            # Número de observaciones del problema
            n_obs = len(Y)
            mdl = Model("FDH INPUT ORIENTED RUSSELL")
            # Variables
            # Variable theta
            name_theta = range(0, n_inp)
            theta_var = mdl.continuous_var_dict(name_theta, name="theta", ub=1, lb=0)
            # Variable landa
            name_landa = range(0, n_obs)
            landa_var = mdl.continuous_var_dict(name_landa, name="landa")
            # Función objetivo
            mdl.minimize(mdl.sum(theta_var[inp] for inp in range(n_inp)) / n_inp)
            # Restricciones
            for inp in range(n_inp):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * x[obs][inp] for obs in range(n_obs)) <= theta_var[inp] * x[obs][inp]
                )
            for r in range(n_out):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * y[obs][r] for obs in range(n_obs)) >= y[obs][r]
                )
            mdl.add_constraint(mdl.sum(landa_var[obs] for obs in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = mdl.solution.get_objective_value()
            else:
                eff = 0
            list_eff.append(eff)
        return list_eff

    def calculate_dea_ruo(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_inp = len(X.columns)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_out = len(Y.columns)
            # Número de observaciones del problema
            n_obs = len(Y)
            mdl = Model("DEA OUTPUT ORIENTED RUSSELL")
            # Variables
            # Variable phi
            name_phi = range(0, n_out)
            phi_var = mdl.continuous_var_dict(name_phi, name="phi", ub=1e33, lb=1)
            # Variable landa
            name_landa = range(0, n_obs)
            landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
            # Función objetivo
            mdl.maximize(mdl.sum(phi_var[r] for r in range(n_out)) / n_out)
            # Restricciones
            for inp in range(n_inp):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * x[obs][inp] for obs in range(n_obs)) <= x[obs][inp]
                )
            for r in range(n_out):
                mdl.add_constraint(
                    mdl.sum(landa_var[obs] * y[obs][r] for obs in range(n_obs)) >= y[obs][r] * phi_var[r]
                )
            mdl.add_constraint(mdl.sum(landa_var[obs] for obs in range(n_obs)) == 1)
            msol = mdl.solve()
            if msol is not None:
                eff = mdl.solution.get_objective_value()
            else:
                eff = 0
            # print(mdl.export_to_string())
            list_eff.append(eff)
        return list_eff

    def calculate_dea_erg(self):
        list_eff = list()
        for obs in range(len(self.data)):
            # Datos de las variables distintas de Y
            X = self.data.filter(self.inputs)
            x = X.values.tolist()
            # Número de dimensiones X del problema
            n_inp = len(X.columns)
            # Datos de las variables Y
            Y = self.data.filter(self.outputs)
            y = Y.values.tolist()
            # Número de dimensiones y del problema
            n_out = len(Y.columns)
            # Número de observaciones del problema
            n_obs = len(Y)

            mdl = Model("DEA ENHANCED RUSSELL GRAPH")
            # Variables

            # Variable beta
            beta_var = mdl.continuous_var(name="beta", ub=1e33, lb=0)

            # Variable t-
            name_t_neg = range(0, n_inp)
            t_neg_var = mdl.continuous_var_dict(name_t_neg, name="t_neg", ub=1e33, lb=0)

            # Variable t+
            name_t_pos = range(0, n_out)
            t_pos_var = mdl.continuous_var_dict(name_t_pos, name="t_pos", ub=1e33, lb=0)

            # Variable landa
            name_landa = range(0, n_obs)
            landa_var = mdl.continuous_var_dict(name_landa, name="landa")

            # Función objetivo
            summa = mdl.sum(t_neg_var[inp]/x[obs][inp] for inp in range(n_inp))
            mdl.minimize(beta_var - (1/n_inp) * summa)

            # Restricciones

            # R1
            summa1 = mdl.sum(t_pos_var[r]/y[obs][r] for r in range(n_out))
            mdl.add_constraint(
                beta_var + (1/n_out) * summa1 == 1
            )

            # R2
            for inp in range(n_inp):
                mdl.add_constraint(
                    -beta_var * x[obs][inp] +
                    mdl.sum(landa_var[obs] * x[obs][inp] for obs in range(n_obs)) +
                    t_neg_var[inp]
                    == 0
                )

            # R3
            for r in range(n_out):
                mdl.add_constraint(
                    -beta_var * y[obs][r] +
                    mdl.sum(landa_var[obs] * y[obs][r] for obs in range(n_obs)) -
                    t_pos_var[r]
                    == 0
                )

            # R4
            mdl.add_constraint(mdl.sum(landa_var[obs] for obs in range(n_obs)) == beta_var)

            msol = mdl.solve()
            if msol is not None:
                eff = mdl.solution.get_objective_value()
            else:
                eff = 0
            # print(mdl.export_to_string())
            list_eff.append(eff)
        return list_eff

    def get_efficiencies(self):
        self.df_eff = self.data.copy()
        switch_method = {
            "ri": self.calculate_dea_ri(),
            "ro": self.calculate_dea_ro(),
            "ddf": self.calculate_dea_ddf(),
            "wa": self.calculate_dea_wa(),
            "rui": self.calculate_dea_rui(),
            "ruo": self.calculate_dea_ruo(),
            "erg": self.calculate_dea_erg()
        }
        for met in self.methods:
            self.df_eff[met] = switch_method.get(met)

    def calculate_wa_w_inp(self):
        X = self.data.filter(self.inputs)
        X = X.to_numpy()
        n_inp = len(self.inputs)
        n_out = len(self.outputs)
        w_inp = list()
        for i in range(n_inp):
            ran = (max(X[:, i]) - min(X[:, i]))
            summa = (n_inp + n_out)
            w = 1 / (summa * ran)
            w_inp.append(w)
        return w_inp

    def calculate_wa_w_out(self):
        Y = self.data.filter(self.outputs)
        Y = Y.to_numpy()
        w_out = list()
        n_inp = len(self.inputs)
        n_out = len(self.outputs)
        for i in range(n_out):
            ran = (max(Y[:, i]) - min(Y[:, i]))
            summa = (n_inp + n_out)
            w = 1 / (summa * ran)
            w_out.append(w)
        return w_out
