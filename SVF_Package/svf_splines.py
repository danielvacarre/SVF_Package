from docplex.mp.model import Model
from svf_package.grid.svf_splines_grid import SVF_SPLINES_GRID
from svf_package.svf import SVF
from svf_package.svf_solution import SVFSolution

class SVF_SP(SVF):
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
        """Metodo que entrena un modelo SVF Splines
        """

        y_df = self.data.filter(self.outputs)
        y = y_df.values.tolist()

        # Numero de dimensiones y del problema
        n_out = len(y_df.columns)
        # Numero de observaciones del problema
        n_obs = len(y)
        n_inp= len(self.outputs)

        #######################################################################
        # Crear el grid
        self.grid = SVF_SPLINES_GRID(self.data, self.inputs, self.d)
        self.grid.create_grid()

        # Número de variables w
        n_var = 0
        for i in range(len(self.grid.df_grid["phi"][0])):
            n_var += len(self.grid.df_grid["phi"][i])

        # Variable w
        # name_w: (i,j)-> i:es el indice de la columna de la matriz phi;j: es el índice de la dimensión de y
        name_w = [(i, j, r) for r in range(n_out) for j in range(0, n_inp) for i in range(0, len(self.grid.knot_list[j]) + 1)]
        w = {}
        w = w.fromkeys(name_w, 1)
        # print(w)
        #
        # Variable Xi
        name_xi = [(i, j) for i in range(0, n_obs) for j in range(0, n_out)]
        xi = {}
        xi = xi.fromkeys(name_xi, self.C)
        
        mdl = Model("SVF Splines")
        mdl.context.cplex_parameters.threads = 1
    
        # Variable w
        w_var = mdl.continuous_var_dict(name_w, ub=1e+33, lb=-1e+33, name='w')
        # Variable xi
        xi_var = mdl.continuous_var_dict(name_xi, ub=1e+33, lb=0, name='xi')
    
        # Función objetivo
        mdl.minimize(mdl.sum((w_var[i] * w[i]) ** 2 for i in name_w) + mdl.sum(xi_var[i] * xi[i] for i in name_xi))
    
        # Restricciones
        for obs in range(0, n_obs):
            for r in range(0, n_out):
                left_side = -y[obs][r] + mdl.sum(
                    w_var[i, j, r] * self.grid.df_grid.phi[obs][j][i] for j in range(0, n_inp) for i in range(0, len(self.grid.knot_list[j]) + 1))
                # (1)
                mdl.add_constraint(
                    left_side <= self.eps + xi_var[obs, r],
                    ctname='c1_o' + str(obs + 1) + "_y" + str(r + 1)
                )
                # (2)
                mdl.add_constraint(
                    -left_side <= 0,
                    ctname='c2_o' + str(obs + 1) + "_y" + str(r + 1)
                )
    
        for r in range(n_out):
            for j in range(n_inp):
                for k in range(2, len(self.grid.knot_list[j]) + 2):
                    left_side = mdl.sum(w_var[i, j, r] * 1 for i in range(1, k))
                    # (3)
                    mdl.add_constraint(
                        left_side >= 0,
                        ctname='c3_x' + str(j + 1) + '_y' + str(r + 1)
                    )
    
        for r in range(n_out):
            for j in range(0, n_inp):
                # for i in range(0, len(t[j]) + 1):
                for i in range(1, len(self.grid.knot_list[j]) + 1):
                    # (4)
                    if i <= 1:
                        mdl.add_constraint(
                            w_var[i, j, r] >= 0,
                            'c4_x' + str(j + 1) + "_y" + str(r + 1)
                        )
                    else:
                        mdl.add_constraint(
                            w_var[i, j, r] <= 0,
                            'c4_x' + str(j + 1) + "_y" + str(r + 1)
                        )
    
        # print(mdl.export_to_string())
        self.model = mdl

    def modify_model(self, c, eps):
        """Método que se utiliza para modificar el valor de C y las restricciones de un modelo
        Args:
            c (float): Valores del hiperparámetro C del modelo_
            eps (float): Valores del hiperparámetro épsilon del modelo

        Returns:
            docplex.mp.model.Model: modelo SVF modificado
        """
        n_out = len(self.outputs)
        n_dmu = len(self.data)
        model = self.model.copy()
        name_var = model.iter_variables()
        name_w = list()
        name_xi = list()
        for var in name_var:
            name = var.get_name()
            if name.find("w") == -1:
                name_xi.append(name)
            else:
                name_w.append(name)
        # Variable w
        w = {}
        w = w.fromkeys(name_w, 1)
        # Variable Xi
        xi = {}
        xi = xi.fromkeys(name_xi, c)
        a = [(model.get_var_by_name(i) * w[i]) ** 2 for i in name_w]

        b = [model.get_var_by_name(i) * xi[i] for i in name_xi]
        # Función objetivo
        model.minimize(model.sum(a) + model.sum(b))
        # Modificar restricciones
        for j in range(0, n_out):
            for i in range(0, n_dmu):
                const_name = 'c1_o' + str(i + 1) + "_y" + str(j + 1)
                rest = model.get_constraint_by_name(const_name)
                rest.rhs += eps
        return model

    def solve(self):
        """Solución de un modelo SVF
        """
        n_inp = self.inputs
        n_out = self.outputs
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
        # Numero de ws por dimensión
        n_w_dim = int(len(sol_w) / n_out / n_inp)
        mat_w = []
        cont = 0
        for i in range(0, n_out):
            mat_2 = []
            for j in range(0, n_inp):
                mat_3 = []
                for r in range(n_w_dim):
                    mat_3.append(round(sol_w[cont], 6))
                    cont += 1
                mat_2.append(mat_3)
            mat_w.append(mat_2)
        self.solution = SVFSolution(mat_w, sol_xi)