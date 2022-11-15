class CrossValidation(object):
    """Objeto representa una validación cruzada

        :param method: Método SVF que se quiere utilizar
        :param inputs: Inputs a evaluar en el conjunto de datos
        :param outputs: Outputs a evaluar en el conjunto de datos
        :param data: Conjunto de datos a evaluar
        :param C: Valores del hiperparámetro C que queremos evaluar
        :param eps: Valores del hiperparámetro épsilon que queremos evaluar
        :param d: Valores del hiperparámetro d que queremos evaluar
        :param seed: Semilla aleatoria para realizar la validación cruzada
        :param n_folds: Número de folds del método de validación cruzada (<=1, indica que se aplica un train-test de 80%
                        train-20%test,>2, indica que se aplican n_folds)
        :param verbose: Indica si se quiere mostrar por pantalla los registros de la validación cruzada


        :type method: str
        :type inputs: list
        :type data: DataFrame
        :type C: list
        :type eps: list
        :type d: list
        :type seed: int
        :type n_folds: int, optional
        :type verbose: bool, optional
    """

    def __init__(self, method, inputs, outputs, data, C, eps, d, seed=0, n_folds=0, verbose=False):
        """
            Método constructor de la validación cruzada

            :param method: Método SVF que se quiere utilizar
            :param inputs: Inputs a evaluar en el conjunto de datos
            :param outputs: Outputs a evaluar en el conjunto de datos
            :param data: Conjunto de datos a evaluar
            :param C: Valores del hiperparámetro C que queremos evaluar
            :param eps: Valores del hiperparámetro épsilon que queremos evaluar
            :param d: Valores del hiperparámetro d que queremos evaluar
            :param seed: Semilla aleatoria para realizar la validación cruzada
            :param n_folds: Número de folds del método de validación cruzada (<=1, indica que se aplica un train-test de 80%train-20%test,>2, indica que se aplican n_folds)
            :param verbose: Indica si se quiere mostrar por pantalla los registros de la validación cruzada
            :param results = None
            self.results_by_fold = None
            self.folds = None
            self.best_C = None
            self.best_eps = None
            self.best_d = None

            :type method: str
            :type inputs: list
            :type outputs: list
            :type data: pandas.DataFrame
            :type C: list
            :type eps: list
            :type d: list
            :type seed: int, optional
            :type n_folds: int, optional
            :type verbose: bool, optional
        """

        self.method = method
        self.inputs = inputs
        self.outputs = outputs
        self.data = data
        self.C = C
        self.eps = eps
        self.d = d
        self.seed = seed
        self.n_folds = n_folds
        self.verbose = verbose
        self.results = None
        self.results_by_fold = None
        self.folds = None
        self.best_C = None
        self.best_eps = None
        self.best_d = None
