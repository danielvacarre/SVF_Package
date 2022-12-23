from datetime import datetime

from docplex.mp.model import Model

from svf_package.grid.svf_splines_grid import SVFSplinesGrid
from svf_package.methods.svf import SVF
from svf_package.solution.svf_solution import SVFPrimalSolution

FMT = "%d-%m-%Y %H:%M:%S"


class SVFSplines(SVF):
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

        now = datetime.now()
        inicio_train = now.strftime(FMT)

        y_df = self.data.filter(self.outputs)
        y = y_df.values.tolist()

        # Numero de dimensiones y del problema
        n_out = len(y_df.columns)
        # Numero de observaciones del problema
        n_obs = len(y)
        n_inp = len(self.inputs)

        #######################################################################
        # Crear el grid
        self.grid = SVFSplinesGrid(self.data, self.inputs, self.outputs, self.d)
        self.grid.create_grid()

        # Número de variables w
        n_var = 0
        for i in range(len(self.grid.data_grid["phi"][0][0])):
            var_dim = self.grid.data_grid["phi"][0][0][i]
            n_var += len(var_dim)
        # Variable w
        name_w = [(knot, inp, out) for out in range(n_out) for inp in range(n_inp) for knot in
                  range(len(self.grid.knot_list[inp]) + 1)]

        w = {}
        w = w.fromkeys(name_w, 1)
        # Variable Xi
        name_xi = [(obs, out) for obs in range(n_obs) for out in range(n_out)]
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
        for obs in range(n_obs):
            for out in range(n_out):
                left_side = -y[obs][out] + mdl.sum(
                    w_var[knot, inp, out] * self.grid.data_grid.phi[obs][out][inp][knot] for inp in range(n_inp) for
                    knot in range(len(self.grid.knot_list[inp]) + 1))
                # (1)
                mdl.add_constraint(
                    left_side <= self.eps + xi_var[obs, out],
                    ctname='c1_o' + str(obs + 1) + "_y" + str(out + 1)
                )
                # (2)
                mdl.add_constraint(
                    -left_side <= 0,
                    ctname='c2_o' + str(obs + 1) + "_y" + str(out + 1)
                )

        for out in range(n_out):
            #     left_side = w_var[0, 0, 0]
            #     mdl.add_constraint(
            #         left_side >= 0,
            #         ctname='c3_x' + str(1) + '_y' + str(out + 1)
            #     )
            for inp in range(n_inp):
                for knot in range(2, len(self.grid.knot_list[inp]) + 2):
                    left_side = mdl.sum(w_var[knot, inp, out] * 1 for knot in range(1, knot))
                    # (3)
                    mdl.add_constraint(
                        left_side >= 0,
                        ctname='c3_x' + str(inp + 1) + '_y' + str(out + 1)
                    )

        for out in range(n_out):
            for inp in range(n_inp):
                # for i in range(len(t[inp]) + 1):
                for knot in range(1, len(self.grid.knot_list[inp]) + 1):
                    # (4)
                    if knot <= 1:
                        mdl.add_constraint(
                            w_var[knot, inp, out] >= 0,
                            'c4_x' + str(inp + 1) + "_y" + str(out + 1)
                        )
                    else:
                        mdl.add_constraint(
                            w_var[knot, inp, out] <= 0,
                            'c4_x' + str(inp + 1) + "_y" + str(out + 1)
                        )

        self.model = mdl
        if self.model_d is None:
            self.model_d = mdl
        now = datetime.now()
        fin_train = now.strftime(FMT)
        self.train_time = datetime.strptime(fin_train, FMT) - datetime.strptime(inicio_train, FMT)

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
        model = self.model_d.copy()
        model.name = "SVF,C:" + str(c) + ",eps:" + str(eps) + ",d:" + str(self.d)
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
        for out in range(n_out):
            for dmu in range(n_dmu):
                const_name = 'c1_o' + str(dmu + 1) + "_y" + str(out + 1)
                rest = model.get_constraint_by_name(const_name)
                rest.rhs += eps
        return model

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
        mat_w = [[] for _ in range(n_out)]
        cont = 0
        for i in range(n_out):
            for j in range(len(self.grid.data_grid["phi"][i][0][0])):
                w = round(sol_w[cont], 6)
                mat_w[i].append(w)
                cont += 1

        mat_xi = [[] for _ in range(n_out)]
        cont = 0
        for i in range(n_out):
            for j in range(len(self.data)):
                mat_xi[i].append(round(sol_xi[cont], 6))
                cont += 1

        self.solution = SVFPrimalSolution(mat_w, sol_xi)
        now = datetime.now()
        fin_solve = now.strftime(FMT)
        self.solve_time = datetime.strptime(fin_solve, FMT) - datetime.strptime(inicio_solve, FMT)

    def get_estimation(self, dmu):
        """Estimacion de una DMU escogida. y=phi(x)*w
        Args:
            dmu (list): Observación sobre la que estimar su valor
        Returns:
            list: Devuelve una lista con la estimación de cada output
        """
        if len(dmu) != len(self.inputs):
            raise RuntimeError("El número de inputs de la DMU no coincide con el número de inputs del problema.")
        phi = self.grid.calculate_dmu_phi(dmu)
        prediction_list = list()
        for out in range(len(self.outputs)):
            # print(self.solution.w[out], phi[out])
            prediction = round(sum([a * b for a, b in zip(self.solution.w[out], phi[out][0])]), 3)
            prediction_list.append(prediction)
        return prediction_list
