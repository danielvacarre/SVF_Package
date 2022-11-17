from numpy import arange
from SVF_Methods.SSVF import SSVF
from SVF_Methods.SVFSolution import SVFSolution


class SVF:
    """
        Clase del algoritmo Support Vector Frontiers
    """

    def __init__(self, method, inputs, outputs, data, C, eps, d):
        """
            Constructor de la clase SVF_Methods
        Args:
            method (string): Método SVF_Methods que se quiere utilizar
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
        self.t = None
        self.vector_subind = None
        self.matrix_phi = None
        self.model = None
        self.solution = None
        self.name = None
        self.model_d = None

    def train(self):
        """
            Método para entrenar el modelo generado. Se selecciona el algoritmo seleccionado y se entrena.
        """
        if self.method == "SVF-SP":
            print("SVF-SP")
            #self.train_svf_sp()
        elif self.method == "SSVF":
            print("SSVF")
            ssvf = SSVF(self.inputs,self.outputs,self.data, self.C, self.eps, self.d)
            model = ssvf.train_ssvf()
        elif self.method == "SVF":
            print("SVF")
            #self.train_svf()
        else:
            raise RuntimeError("The method selected doesn't exist")
        return model

    #FUNCIONES DEL TRAIN

    def modify_model(self, c, eps):
        n_obs = len(self.data)
        model = self.model_d.copy()
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

        a = [model.get_var_by_name(i) * model.get_var_by_name(i) * w[i] for i in name_w]
        b = [model.get_var_by_name(i) * xi[i] for i in name_xi]
        # Funcion objetivo
        model.minimize(model.sum(a) + model.sum(b))
        # Modificar restricciones
        for i in range(0, n_obs):
            for r in range(len(self.outputs)):
                const_name = 'c2_' + str(i) + "_" + str(r)
                rest = model.get_constraint_by_name(const_name)
                rest.rhs += eps
        return model
            
    def solve(self):
        n_dim_y = len(self.outputs)
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
        n_w_dim = int(len(sol_w) / n_dim_y)
        mat_w = [[] for _ in range(0, n_dim_y)]
        cont = 0
        for i in range(0, n_dim_y):
            for j in range(0, n_w_dim):
                mat_w[i].append(round(sol_w[cont], 6))
                cont += 1
        return SVFSolution(mat_w, sol_xi)