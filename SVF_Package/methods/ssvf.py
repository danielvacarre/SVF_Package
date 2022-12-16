from datetime import datetime

from docplex.mp.model import Model

from svf_package.grid.svfgrid import SVFGrid
from svf_package.methods.svf import SVF
from svf_package.solution.svf_solution import SVFPrimalSolution

FMT = "%d-%m-%Y %H:%M:%S"


class SSVF(SVF):
    """Clase del modelo SVF Simplificado
    """

    def __init__(self, method, inputs, outputs, data, C, eps, d):
        """Constructor de la clase SSVF

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
        """Metodo que entrena un modelo SSVF
        """

        now = datetime.now()
        inicio_train = now.strftime(FMT)

        y_df = self.data.filter(self.outputs)
        y = y_df.values.tolist()

        # Numero de dimensiones y del problema
        n_out = len(y_df.columns)
        # Numero de observaciones del problema
        n_obs = len(y)

        #######################################################################
        # Crear el grid
        self.grid = SVFGrid(self.data, self.inputs, self.outputs, self.d)
        self.grid.create_grid()

        # Numero de variables w
        n_var = len(self.grid.data_grid.phi[0][0])

        #######################################################################

        # Variable w
        # name_w: (i,j)-> i:es el indice de la columna de la matriz phi;j: es el indice de la dimension de y
        name_w = [(out, w_var) for out in range(n_out) for w_var in range(n_var)]
        w = {}
        w = w.fromkeys(name_w, 1)

        # Variable Xi
        name_xi = [(out, obs) for out in range(n_out) for obs in range(n_obs)]
        xi = {}
        xi = xi.fromkeys(name_xi, self.C)

        mdl = Model("SSVF C:" + str(self.C) + ", eps:" + str(self.eps) + ", d:" + str(self.d))
        mdl.context.cplex_parameters.threads = 1

        # Variable w
        w_var = mdl.continuous_var_dict(name_w, ub=1e+33, lb=0, name='w')
        # Variable xi
        xi_var = mdl.continuous_var_dict(name_xi, ub=1e+33, lb=0, name='xi')

        # Funcion objetivo
        mdl.minimize(mdl.sum(w_var[i] * w[i] for i in name_w) + mdl.sum(xi_var[i] * xi[i] for i in name_xi))

        # Restricciones
        for obs in range(n_obs):
            for out in range(n_out):
                left_side = y[obs][out] - mdl.sum(w_var[out, var] * self.grid.data_grid.phi[obs][out][var]
                                                  for var in range(n_var))
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
        self.model = mdl
        if self.model_d is None:
            self.model_d = mdl

        now = datetime.now()
        fin_train = now.strftime(FMT)
        self.train_time = datetime.strptime(fin_train, FMT) - datetime.strptime(inicio_train, FMT)

    def solve(self):
        """Solución de un modelo SVF
        """
        now = datetime.now()
        inicio_solve = now.strftime(FMT)

        n_out = len(self.outputs)
        self.model.solve()
        name_var = self.model.iter_variables()
        sol_w = list()
        sol_xi = list()
        for var in name_var:
            name = var.get_name()
            sol = self.model.solution[name]
            if name.find("w") == -1:
                sol_xi.append(sol)
            else:
                sol_w.append(sol)
        # Numero de ws por dimension
        n_w_dim = int(len(sol_w) / n_out)
        mat_w = [[] for _ in range(n_out)]
        cont = 0
        for out in range(n_out):
            for j in range(0, n_w_dim):
                mat_w[out].append(round(sol_w[cont], 6))
                cont += 1
        mat_xi = [[] for _ in range(n_out)]
        cont = 0
        for out in range(n_out):
            for j in range(0, len(self.data)):
                mat_xi[out].append(round(sol_xi[cont], 6))
                cont += 1
        self.solution = SVFPrimalSolution(mat_w, mat_xi)

        now = datetime.now()
        fin_solve = now.strftime(FMT)
        self.solve_time = datetime.strptime(fin_solve, FMT) - datetime.strptime(inicio_solve, FMT)