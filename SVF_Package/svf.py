from svf_package.svf_solution import SVFSolution

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



    def estimation(self,x):
        """Estimacion de una DMU escogida. y=phi(x)*w

        Args:
            x (list): Observación sobre la que estimar su valor

        Returns:
            list: Devuelve una lista con la estimación de cada output
        """
        p = self.grid.search_observation(x)
        phi = self.grid.calculate_phi_observation(p)
        prediction_list = list()
        prediction = 0
        for i in range(0, len(self.outputs)):
            for j in range (len(self.solution.w)):
                prediction += self.solution.w[i][j] * phi[i]
            prediction_list.append(prediction)
        return prediction_list