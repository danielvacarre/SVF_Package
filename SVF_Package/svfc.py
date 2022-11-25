from docplex.mp.model import Model
from numpy import asarray, float32
from svf_package.grid.svf_grid import SVF_GRID
from svf_package.svf import SVF
from svf_package.svf_solution import SVFSolution

class SVFC(SVF):
    """Clase del modelo SVF Splines
    """

    def __init__(self, method, inputs, outputs, data, C, eps, d):
        """Constructor de la clase SVF Splines

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

        y_df = self.data.filter(self.outputs)
        y = y_df.values.tolist()

        # Numero de dimensiones y del problema
        n_out = len(y_df.columns)
        # Numero de observaciones del problema
        n_obs = len(y)

        # Crear el grid
        self.grid = SVF_GRID(self.data, self.inputs, self.outputs, self.d)
        self.grid.create_grid()

        # Numero de variables w
        n_var = len(self.grid.data_grid.phi[0][0])

        # Variable u
        name_u = [(out, var) for out in range(n_out) for var in range(n_var)]
        u = {}
        u = u.fromkeys(name_u, 1)
        # Variable u
        name_v = [(out, var) for out in range(n_out) for var in range(n_var)]
        v = {}
        v = v.fromkeys(name_v, 1)
        # Variable Xi
        name_xi = [(out, var) for out in range(n_out) for var in range(n_obs)]
        xi = {}
        xi = xi.fromkeys(name_xi, self.C)
        mdl = Model("SVF C:" + str(self.C) + ", eps:" + str(self.eps) + ", d:" + str(self.d))
        mdl.context.cplex_parameters.threads = 1

        # Variable w
        u_var = mdl.continuous_var_dict(name_u, ub=1e+33, lb=0, name='u')
        v_var = mdl.continuous_var_dict(name_u, ub=1e+33, lb=0, name='v')
        # Variable xi
        xi_var = mdl.continuous_var_dict(name_xi, ub=1e+33, lb=0, name='xi')

        # Funcion objetivo
        mdl.minimize(mdl.sum(u_var[i] * u[i] for i in name_u) +
                     mdl.sum(v_var[i] * v[i] for i in name_v) +
                     mdl.sum(xi_var[i] * xi[i] for i in name_xi))
        # Restricciones
        for obs in range(n_obs):
            for out in range(n_out):
                left_side = y[obs][out] - \
                            mdl.sum(
                                u_var[out, var] * self.grid.data_grid.phi[obs][out][var] -
                                v_var[out, var] * self.grid.data_grid.phi[obs][out][var] for var in range(n_var)
                            )
                # (1)
                mdl.add_constraint(
                    left_side <= 0,
                    ctname='c1_' + str(obs) + "_" + str(out)
                )
                # (2)
                mdl.add_constraint(
                    -left_side <= self.eps + xi_var[out, obs],
                    ctname='c2_' + str(obs) + "_" + str(out)
                )
        # (3)
        for out in range(n_out):
            lhs = mdl.sum(
                u_var[out, var] * self.grid.df_grid.phi[0][out][var] - v_var[out, var] * self.grid.df_grid.phi[0][out][var] for var in
                range(n_var)
            )
            mdl.add_constraint(
                lhs >= 0,
                ctname='c3_' + str(self.grid.df_grid.id_cell[0]) + "_" + str(out)
            )
        for index, cell in self.grid.df_grid.iterrows():
            left_side = cell["phi"][0]
            c_cont = cell["c_cells"]
            for c_cell in c_cont:
                c_cont_row = self.grid.df_grid.loc[self.grid.df_grid['id_cell'] == c_cell]
                right_side = c_cont_row["phi"].values[0][0]
                constraint = asarray(left_side, dtype=float32) - asarray(right_side, dtype=float32)
                for out in range(n_out):
                    lhs = mdl.sum(
                        u_var[out, var] * constraint[var] - v_var[out, var] * constraint[var] for var in range(n_var)
                    )
                    mdl.add_constraint(
                        lhs >= 0,
                        ctname='c3_' + str(cell["id_cell"]) + "_" + str(out)
                    )

        self.model = mdl

    def solve(self):
        """Solución de un modelo SVF
        """
        n_out = len(self.outputs)
        self.model.solve()
        name_var = self.model.iter_variables()
        sol_u = list()
        sol_v = list()
        sol_xi = list()
        for var in name_var:
            name = var.get_name()
            sol = self.model.solution[name]
            if name.find("u") == 0:
                sol_u.append(sol)
            elif name.find("v") == 0:
                sol_v.append(sol)
            else:
                sol_xi.append(sol)
        # Numero de ws por dimension
        n_w_dim = int(len(sol_u) / n_out)
        mat_w = [[] for _ in range(n_out)]
        cont = 0
        for out in range(n_out):
            for w_dim in range(n_w_dim):
                w = sol_u[cont] - sol_v[cont]
                mat_w[out].append(w)
                cont += 1
        mat_xi = [[] for _ in range(n_out)]
        cont = 0
        for out in range(n_out):
            for obs in range(len(self.data)):
                mat_xi[out].append(round(sol_xi[cont], 6))
                cont += 1
        self.solution = SVFSolution(mat_w, mat_xi)

    def estimation(self, dmu):
        """Estimacion de una DMU escogida. y=phi(dmu)*w

        Args:
            dmu (list): Observación sobre la que estimar su valor

        Returns:
            list: Devuelve una lista con la estimación de cada output
        """
        dmu_cell = self.grid.search_dmu(dmu)
        phi = self.grid.df_grid.loc[self.grid.df_grid['id_cell'] == dmu_cell,"phi"].values[0]
        prediction_list = list()
        for out in range(len(self.outputs)):
            prediction = round(sum([a * b for a, b in zip(self.solution.w[out], phi[out])]),3)
            prediction_list.append(prediction)
        return prediction_list


