from datetime import datetime

from docplex.mp.model import Model
from numpy import dot, array, around

from svf_package.grid.svfgrid import SVFGrid
from svf_package.methods.svf import SVF
from svf_package.solution.svf_dual_solution import SVFDualSolution

FMT = "%d-%m-%Y %H:%M:%S"


class SVFDual(SVF):

    def __init__(self, method, inputs, outputs, data, C, eps, d):
        """Constructor de la clase SVFDual

        Args:
            method (string): Método SVF que se quiere utilizar
            inputs (list): Inputs a evaluar en el conjunto de dato
            outputs (list): Outputs a evaluar en el conjunto de datos
            data (pandas.DataFrame): Conjunto de datos a evaluar
            C (float): Valores del hiperparámetro C del modelo
            eps (float): Valores del hiperparámetro épsilon del modelo
            d (int): Valor del hiperparámetro d del modelo
        """
        super().__init__(method, inputs, outputs, data, C, eps, d)

    def train(self):
        """Método que entrena un modelo SVF en su forma dual
        """
        now = datetime.now()
        inicio_train = now.strftime(FMT)

        outputs_df = self.data.filter(self.outputs)
        y = outputs_df.values.tolist()

        # Numero de dimensiones y del problema
        n_out = len(outputs_df.columns)
        # Numero de observaciones del problema
        n_obs = len(y)

        #######################################################################
        # Matriz de t y de indices de ts
        # Crear el grid
        self.grid = SVFGrid(self.data, self.inputs, self.outputs, self.d)
        self.grid.create_grid()

        # Numero de variables w
        n_var = len(self.grid.data_grid.phi[0][0])

        #######################################################################

        # Variable alpha
        name_alpha = [(obs, out) for obs in range(0, n_obs) for out in range(0, n_out)]
        alpha = {}
        alpha = alpha.fromkeys(name_alpha, 1)

        # Variable delta
        name_delta = [(obs, out) for obs in range(0, n_obs) for out in range(0, n_out)]
        delta = {}
        delta = delta.fromkeys(name_delta, 1)

        # Variable gamma
        name_gamma = [(var, out) for var in range(0, n_var) for out in range(0, n_out)]

        mdl = Model("SVF DUAL C:" + str(self.C) + ", eps:" + str(self.eps) + ", d:" + str(self.d))
        mdl.context.cplex_parameters.threads = 1

        # Variable alpha
        alpha_var = mdl.continuous_var_dict(name_alpha, ub=1e+33, lb=0, name='alpha')
        # Variable delta
        delta_var = mdl.continuous_var_dict(name_delta, ub=self.C, lb=0, name='delta')
        # Variable gamma
        gamma_var = mdl.continuous_var_dict(name_gamma, ub=1e+33, lb=0, name='gamma')

        # Funcion objetivo
        s1 = mdl.sum(
            (alpha_var[obs, out] * alpha[obs, out] - delta_var[obs, out] * delta[obs, out]) *
            (alpha_var[obs2, out] * alpha[obs2, out] - delta_var[obs2, out] * delta[obs2, out]) *
            (dot(self.grid.data_grid.phi[obs][out], self.grid.data_grid.phi[obs2][out])) for out in range(n_out)
            for obs in range(n_obs) for obs2 in range(n_obs)
        )
        s2 = mdl.sum(
            2 * (alpha_var[obs, out] * alpha[obs, out] - delta_var[obs, out] * delta[obs, out]) *
            mdl.sum(self.grid.data_grid.phi[obs][out][var] * gamma_var[var, out])
            for out in range(n_out) for obs in range(n_obs) for var in range(n_var)
        )
        s3 = mdl.sum(gamma_var[var, out] * gamma_var[var, out] for out in range(n_out) for var in range(n_var))
        s4 = self.eps * mdl.sum(delta_var[obs, out] * delta[obs, out] for out in range(n_out) for obs in range(n_obs))
        s5 = mdl.sum(
            (alpha_var[obs, out] * alpha[obs, out] - delta_var[obs, out] * delta[obs, out]) * y[obs][out] for out in
            range(n_out) for obs in
            range(n_obs))

        mdl.minimize(1 / 2 * (s1 + s2 + s3) + s4 - s5)

        self.model = mdl

        now = datetime.now()
        fin_train = now.strftime(FMT)
        self.train_time = datetime.strptime(fin_train, FMT) - datetime.strptime(inicio_train, FMT)

    def solve(self):
        """Método que soluciona el modelo entrenado.
        """
        now = datetime.now()
        inicio_solve = now.strftime(FMT)

        n_obs = len(self.data)
        n_var = len(self.grid.data_grid.phi[0][0])
        n_out = len(self.outputs)
        self.model.solve()

        name_var = self.model.iter_variables()
        sol_gamma = list()
        sol_alpha = list()
        sol_delta = list()
        for var in name_var:
            name = var.get_name()
            sol = self.model.solution[name]
            if name.find("gamma") == 0:
                sol_gamma.append(sol)
            elif name.find("alpha") == 0:
                sol_alpha.append(sol)
            elif name.find("delta") == 0:
                sol_delta.append(sol)

        mat_sol_gamma = [[] for _ in range(0, n_out)]
        mat_sol_alpha = [[] for _ in range(0, n_out)]
        mat_sol_delta = [[] for _ in range(0, n_out)]

        cont = 0
        cont2 = 0
        for out in range(n_out):
            for var in range(n_var):
                mat_sol_gamma[out].append(round(sol_gamma[cont], 6))
                cont += 1
            for k in range(n_obs):
                mat_sol_alpha[out].append(round(sol_alpha[cont2], 6))
                mat_sol_delta[out].append(round(sol_delta[cont2], 6))
                cont2 += 1

        sumatorio = 0
        mat_sol_w = [[] for _ in range(0, n_out)]
        for out in range(n_out):
            for obs in range(n_obs):
                sumatorio += (mat_sol_alpha[out][obs] - mat_sol_delta[out][obs]) * array(
                    self.grid.data_grid.phi[obs][out])
            sol = sumatorio + array(mat_sol_gamma[out])
            mat_sol_w[out] = around(sol, 3).tolist()

        self.solution = SVFDualSolution(mat_sol_gamma, mat_sol_alpha, mat_sol_delta, mat_sol_w)

        now = datetime.now()
        fin_solve = now.strftime(FMT)
        self.solve_time = datetime.strptime(fin_solve, FMT) - datetime.strptime(inicio_solve, FMT)

    def calculate_matrix_transformations_without_l(self, l):
        """Función para el RFE-SVF. Calcula la matriz de phi cuando se elimina el input "l"
        Args:
            l (int): Columna a eliminar dentro del dataset
        Returns:
            pandas.DataFrame: dataframe con las transformaciones de las DMUs sin  el input "l"
        """
        matrix_phi_l = list()
        for dmu_pos in self.grid.data_grid.pos:
            phi = self.calculate_transformation_observation_without_l(dmu_pos, l)
            matrix_phi_l.append(phi)
        return matrix_phi_l

    def calculate_transformation_observation_without_l(self, dmu_pos, l):
        """Función que calula el valor de phi sin el input "l"

        Args:
            dmu_pos (list): posición de la DMU en el grid
            l (int): indice de la columna a eliminar

        Returns:
            list: Devuelve el valor de la transformación de la DMU cuando se elimina el input "l"
        """
        phi = list()
        for celda in self.grid.df_grid.id_cell:
            r = 0
            if celda[l] == 0:
                for j in range(len(celda)):
                    if dmu_pos[j] >= celda[j]:
                        r = 1
                    else:
                        r = 0
                        break
            else:
                r = 0
            phi.append(r)
        return phi
