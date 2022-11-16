from itertools import zip_longest, product
from numpy import arange
from docplex.mp.model import Model

class SSVF():

    def __init__(self, inputs, outputs, data, C, eps, d):
        self.data = data
        self.outputs = outputs
        self.inputs = inputs
        self.C = C
        self.eps = eps
        self.d = d

    def create_matrix_partitions(self,d):

        x = self.data.filter(self.inputs)

        # Numero de columnas x
        n_dim = len(x.columns)
        # Lista de listas de ts
        t = list()
        # Lista de indices (posiciones) para crear el vector de subind
        t_ind = list()
        for col in range(0, n_dim):
            # Ts de la dimension col
            ts = list()
            t_max = x.iloc[:, col].max()
            t_min = x.iloc[:, col].min()
            amplitud = (t_max - t_min) / d
            for i in range(0, d + 1):
                t_i = t_min + i * amplitud
                ts.append(t_i)
            t.append(ts)
            t_ind.append(arange(0, len(ts)))
        return t, t_ind

    def calculate_matrix_transformations(self):
        x = self.data.filter(self.inputs)
        x_list = x.values.tolist()
        M = []
        for x in x_list:
            p = self.locate_position_observation(x)
            phi = self.calculate_transformation_observation(p)
            M.append(phi)
        return M

    def locate_position_observation(self, x):
        p = []
        # transpuesta de t para calcular el vector de posiciones de las observaciones
        r = list(zip_longest(*self.t))
        for l in range(0, len(self.t)):
            for m in range(0, len(self.t[l])):
                trans = self.transformation(x[l], r[m][l])
                if trans < 0:
                    p.append(m - 1)
                    break
                if trans == 0:
                    p.append(m)
                    break
                if trans > 0 and m == len(self.t[l]) - 1:
                    p.append(m)
                    break
        return p

    def calculate_transformation_observation(self, p):
        phi = []
        n_dim = len(p)
        for i in range(0, len(self.vector_subind)):
            for j in range(0, n_dim):
                if p[j] >= self.vector_subind[i][j]:
                    r = 1
                else:
                    r = 0
                    break
            phi.append(r)
        return phi

    def train_ssvf(self):

        y_df = self.data.filter(self.outputs)
        y = y_df.values.tolist()

        # Numero de dimensiones y del problema
        n_dim_y = len(y_df.columns)
        # Numero de observaciones del problema
        n_obs = len(y)

        #######################################################################
        # Matriz de t y de indices de ts

        self.t, t_ind = self.create_matrix_partitions(self.d)
        # vector de subindices de w
        self.vector_subind = list()
        for combination in product(*t_ind):
            self.vector_subind.append(combination)
        self.matrix_phi = self.calculate_matrix_transformations()

        # Numero de variables w
        n_var = len(self.matrix_phi[0])
        #######################################################################

        # Variable w
        # name_w: (i,j)-> i:es el indice de la columna de la matriz phi;j: es el indice de la dimension de y
        name_w = [(i, j) for i in range(0, n_dim_y) for j in range(0, n_var)]
        w = {}
        w = w.fromkeys(name_w, 1)

        # Variable Xi
        name_xi = [(i, j) for i in range(0, n_dim_y) for j in range(0, n_obs)]
        xi = {}
        xi = xi.fromkeys(name_xi, self.C)

        mdl = Model("SSVF Multioutput")
        mdl.context.cplex_parameters.threads = 1

        # Variable w
        w_var = mdl.continuous_var_dict(name_w, ub=1e+33, lb=0, name='w')
        # Variable xi
        xi_var = mdl.continuous_var_dict(name_xi, ub=1e+33, lb=0, name='xi')

        # Funcion objetivo
        mdl.minimize(mdl.sum(w_var[i] * w_var[i] * w[i] for i in name_w) + mdl.sum(xi_var[i] * xi[i] for i in name_xi))

        # Restricciones
        for i in range(0, n_obs):
            for dim_y in range(0, n_dim_y):
                left_side = y[i][dim_y] - mdl.sum(w_var[dim_y, j] * self.matrix_phi[i][j] for j in range(0, n_var))
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
        return mdl

    def transformation(self, x_i, t_k):
        """Funcion que evalua si el valor de una celda es mayor o menor al de un nodo del grid.
           Si es mayor devuelve 1, si es igual devuelve 0 y si es menor devuelve -1.

        Parameters
        ----------
        x_i : float
            Valor de la celda a evaluar
        t_k : float
            Valor del nodo con el que se quiere comparar

        Returns
        -------
        res : int
            Resultado de la transformacion
        """

        z = x_i - t_k
        if z < 0:
            return -1
        elif z == 0:
            return 0
        else:
            return 1