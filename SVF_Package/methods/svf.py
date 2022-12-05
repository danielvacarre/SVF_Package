from pandas import DataFrame, concat
from svf_package.efficiency.csvf_eff import CSVFEff
from svf_package.efficiency.dea import DEA
from svf_package.efficiency.fdh import FDH
from svf_package.efficiency.svf_eff import SVFEff


class SVF:
    """Clase del algoritmo Support Vector Frontiers
    """

    def __init__(self, method, inputs, outputs, data, C, eps, d):
        """Constructor de la clase SVF
        Args:
            method (string): Método SVF que se quiere utilizar
            inputs (list): Inputs a evaluar en el conjunto de dato
            outputs (list): Outputs a evaluar en el conjunto de datos
            data (pandas.DataFrame): Conjunto de datos a evaluar
            C (float): Valores del hiperparámetro C del modelo
            eps (float): Valores del hiperparámetro épsilon del modelo
            d (int): Valor del hiperparámetro d del modelo
        """
        self.method = method
        self.data = data
        self.outputs = outputs
        self.inputs = inputs
        self.C = C
        self.eps = eps
        self.d = d
        self.grid = None
        self.model = None
        self.model_d = None
        self.solution = None
        self.name = None
        self.df_estimation = None
        self.df_eff = None
        self.solve_time = None
        self.train_time = None

    def modify_model(self, c, eps):
        """Método que se utiliza para modificar el valor de C y las restricciones de un modelo
        Args:
            c (float): Valores del hiperparámetro C del modelo_
            eps (float): Valores del hiperparámetro épsilon del modelo

        Returns:
            docplex.mp.model.Model: modelo SVF modificado
        """

        n_obs = len(self.data)
        model = self.model_d.copy()
        model.name = "SVF,C:" + str(c) + ",eps:" + str(eps) + ",d:" + str(self.d)
        name_var = model.iter_variables()
        name_w = list()
        name_xi = list()
        for var in name_var:
            name = var.get_name()
            if name.find("xi") == 0:
                name_xi.append(name)
            else:
                name_w.append(name)
        # Variable w
        w = {}
        w = w.fromkeys(name_w, 1)
        # Variable Xi
        xi = {}
        xi = xi.fromkeys(name_xi, c)

        a = [model.get_var_by_name(i) * w[i] for i in name_w]
        b = [model.get_var_by_name(i) * xi[i] for i in name_xi]
        # Funcion objetivo
        model.minimize(model.sum(a) + model.sum(b))
        # Modificar restricciones
        for obs in range(n_obs):
            for r in range(len(self.outputs)):
                const_name = 'c2_' + str(obs) + "_" + str(r)
                rest = model.get_constraint_by_name(const_name)
                rest.rhs += eps
        return model

    def get_estimation(self, dmu):
        """Estimacion de una DMU escogida. y=phi(dmu)*w

        Args:
            dmu (list): Observación sobre la que estimar su valor

        Returns:
            list: Devuelve una lista con la estimación de cada output
        """
        if len(dmu) != len(self.inputs):
            raise RuntimeError("El número de inputs de la DMU no coincide con el número de inputs del problema.")
        dmu_cell = self.grid.search_dmu(dmu)
        phi = self.grid.df_grid.loc[self.grid.df_grid['id_cell'] == dmu_cell, "phi"].values[0]
        prediction_list = list()
        for out in range(len(self.outputs)):
            prediction = round(sum([a * b for a, b in zip(self.solution.w[out], phi[out])]), 3)
            prediction_list.append(prediction)
        return prediction_list

    def get_df_estimation(self):
        if self.solution is None:
            self.solve()
        df_estimation = self.data.filter(self.inputs).copy()
        df_y_est = list()
        for dmu_index in range(len(df_estimation)):
            dmu = df_estimation.iloc[dmu_index].to_list()
            y_est = self.get_estimation(dmu)
            df_y_est.append(y_est)
        name_columns = self.outputs
        df_y_est = DataFrame(df_y_est,columns=name_columns)
        df_y_est = concat((df_estimation, df_y_est), axis=1)
        self.df_estimation = df_y_est

    def get_svf_efficiencies(self,methods):
        if self.df_eff is None:
            self.get_df_estimation()
        eff_method = SVFEff(self.inputs, self.outputs, self.data, methods, self.df_estimation)
        eff_method.get_efficiencies()
        self.df_eff = eff_method.df_eff

    def get_csvf_efficiencies(self,methods):
        if self.df_eff is None:
            self.get_df_estimation()
        eff_method = CSVFEff(self.inputs, self.outputs, self.data, methods, self.df_estimation)
        eff_method.get_efficiencies()
        self.df_eff = eff_method.df_eff

    def get_fdh_efficiencies(self, methods):
        if self.df_eff is None:
            self.get_df_estimation()
        eff_method = FDH(self.inputs, self.outputs, self.data, methods, self.df_estimation)
        eff_method.get_efficiencies()
        return eff_method.df_eff

    def get_dea_efficiencies(self, methods):
        if self.df_eff is None:
            self.get_df_estimation()
        eff_method = DEA(self.inputs, self.outputs, self.data, methods, self.df_estimation)
        eff_method.get_efficiencies()
        return eff_method.df_eff

    def get_df_all_estimation(self):
        df_estimation = self.data.filter(self.inputs).copy()
        df_estimation = df_estimation.join(self.data.filter(self.outputs))
        fdh = self.get_fdh_efficiencies(["ro"])
        df_estimation['y_FDH'] = fdh['ro'] * fdh['y']
        self.get_svf_efficiencies(["ro"])
        df_estimation['y_SVF'] = self.df_eff['ro'] * self.df_eff['y']
        dea = self.get_dea_efficiencies(["ro"])
        df_estimation['y_DEA'] = dea['ro'] * dea['y']
        self.get_csvf_efficiencies(["ro"])
        df_estimation['y_CSVF'] = self.df_eff['ro'] * self.df_eff['y']
        return df_estimation


