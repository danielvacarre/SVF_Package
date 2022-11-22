from docplex.mp.model import Model
from numpy import append, asarray, float32

from svf_package.grid.svf_grid import SVF_GRID
from svf_package.svf import SVF
from cplex import Cplex




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
        n_out= len(self.outputs)

        # Crear el grid
        self.grid = SVF_GRID(self.data, self.inputs, self.d)
        self.grid.create_grid()

        # Numero de variables w
        n_var = len(self.grid.data_grid.phi[0])

        # Variable u
        name_u = [(i, j) for i in range(0, n_out) for j in range(0, n_var)]
        u = {}
        u = u.fromkeys(name_u, 1)
        # Variable u
        name_v = [(i, j) for i in range(0, n_out) for j in range(0, n_var)]
        v = {}
        v = v.fromkeys(name_v, 1)
        # Variable Xi
        name_xi = [(i, j) for i in range(0, n_out) for j in range(0, n_obs)]
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
        for i in range(0, n_obs):
            for dim_y in range(0, n_out):
                left_side = y[i][dim_y] - \
                            mdl.sum(
                                u_var[dim_y, j] * self.grid.data_grid.phi[i][j] -
                                v_var[dim_y, j] * self.grid.data_grid.phi[i][j] for j in range(0, n_var)
                            )
                # (1)
                mdl.add_constraint(
                    left_side <= 0,
                    ctname='c1_' + str(i) + "_" + str(dim_y)
                )
                # (2)
                mdl.add_constraint(
                    -left_side <= self.eps + xi_var[dim_y, i],
                    ctname='c2_' + str(i) + "_" + str(dim_y)
                )
        #(3)
        for index, cell in self.grid.df_grid.iterrows():
            left_side = cell["phi"]
            c_cont = cell["c_cells"]
            for c_cell in c_cont:
                c_cont_row = self.grid.df_grid.loc[self.grid.df_grid['id_cell'] == c_cell]
                right_side = c_cont_row["phi"].values[0]
                constraint = asarray(left_side, dtype=float32) - asarray(right_side, dtype=float32)
                for dim_y in range(0, n_out):
                    lhs = mdl.sum(
                        u_var[dim_y, j] * constraint[j] - v_var[dim_y, j] * constraint[j] for j in range(0, n_var)
                    )
                    mdl.add_constraint(
                        lhs >= 0,
                        ctname='c3_' + str(c_cell) + "_" + str(dim_y)
                    )


        self.model = mdl
        # # Crear el problema de optimización
        # model = Cplex()
        # model.set_log_stream(None)
        # model.set_error_stream(None)
        # model.set_warning_stream(None)
        # model.set_results_stream(None)
        # model.parameters.threads.set(1)
        #
        # # Función objetivo
        # model.objective.set_sense(model.objective.sense.minimize)
        # obj = self.create_obj()
        # # Número de variables u-v + xi del problema
        # n_var = len(obj)
        #
        # # Variables
        # ub = [float(1e33)] * n_var
        # lb = [float(0)] * n_var
        # model.variables.add(obj=obj, ub=ub, lb=lb)
        #
        # self.model = model
        #
        # # Restricciones
        # c3 = self.monotony_constraint()



    def create_obj(self):
        n_var = len(self.grid.data_grid.phi[0]) * 2 # El x2 es porque se pasa a forma u-v
        n_out= len(self.outputs)
        n_obs = len(self.data)
        obj_w = [float(1)] * n_var * n_out
        obj_xi = [float(self.C)] * n_obs * n_out
        obj = append(obj_w, obj_xi)
        return obj

    def monotony_constraint(self, mdl):
        constraint_list = list()
        constraint_list.append(self.grid.data_grid.phi[0])
        for index, cell in self.grid.data_grid.iterrows():
            left_side = cell["phi"]

        return mdl


