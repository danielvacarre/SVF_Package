from pandas import DataFrame
from sklearn.model_selection import KFold

from SVF_Methods.SVF import SVF
from SVF_Methods.FOLD import FOLD


class CrossValidation(object):

    def __init__(self, method, inputs, outputs, data, C, eps, D, seed=0, n_folds=0, verbose=False):
        """Constructor del objeto validació cruzada. Realiza un train-test o una k-folds en base al número de folds seleccionado

        Args:
            method (string): Método SVF_Methods que se quiere utilizar
            inputs (list): Inputs a evaluar en el conjunto de dato
            outputs (list): Outputs a evaluar en el conjunto de datos
            data (pandas.DataFrame): Conjunto de datos a evaluar
            C (list): Valores del hiperparámetro C que queremos evaluar
            eps (list): Valores del hiperparámetro épsilon que queremos evaluar
            D (list): Valores del hiperparámetro d que queremos evaluar
            seed (int, optional): Semilla aleatoria para realizar la validación cruzada. Defaults to 0.
            n_folds (int, optional):Número de folds del método de validación cruzada (<=1, indica que se aplica un train-test de 80%
            train-20%test,>2, indica que se aplican n_folds. Defaults to 0.
            verbose (bool, optional): Indica si se quiere mostrar por pantalla los registros de la validación cruzada. Defaults to False.
        """

        self.method = method
        self.inputs = inputs
        self.outputs = outputs
        self.data = data
        self.C = C
        self.eps = eps
        self.D = D
        self.seed = seed
        self.n_folds = n_folds
        self.verbose = verbose
        self.results = None
        self.results_by_fold = None
        self.folds = None
        self.best_C = None
        self.best_eps = None
        self.best_d = None

    def cv(self):
        """
            Función que ejecuta el tipo de validación cruzada:
                >1: aplica el método k_folds
                
               <=1: aplica el método train-test
        """
        self.results_by_fold = DataFrame(columns=["C", "eps", "d", "error"])
        if self.n_folds > 1:
            self.kfolds()
        else:
            pass
            # self.train_test()

    def kfolds(self):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        fold_num = 0
        list_fold = list()
        for train_index, test_index in kf.split(self.data):
            fold_num += 1
            data_train, data_test = self.data.iloc[train_index], self.data.iloc[test_index]
            fold = FOLD(data_train, data_test, fold_num)
            list_fold.append(fold)
            for d in self.D:
                model = SVF(self.method, self.inputs, self.outputs, self.data, 1, 0, d)
                model.train()
        #         for c in self.C:
        #             for i in range(len(self.eps)):
        #                 if self.verbose == True:
        #                     print("     FOLD:", fold_num, "C:", c, "EPS:", self.eps[i])
        #                 model.C = c
        #                 model.eps = self.eps[i]
        #                 model.d = d
        #                 model.train()
        #                 model.solve()
        #                 # print(deam.solution)
        #                 error_bruto = self.calculate_cv_mse(fold.data_test, model)
        #                 # print(error_bruto)
        #                 self.results_by_fold = self.results_by_fold.append(
        #                     {
        #                         "Num": fold_num,
        #                         "C": c,
        #                         "eps": self.eps[i],
        #                         "error": error_bruto,
        #                     },
        #                     ignore_index=True,
        #                 )
        self.folds = list_fold
        # self.results = self.results_by_fold.groupby(['C', 'eps']).sum() / self.n_folds
        # self.results = self.results.sort_index(ascending=False)
        # self.results = self.results.drop(['Num'], axis=1)
        # min_error = self.results[["error"]].idxmin().values
        # self.best_C = min_error[0][0]
        # self.best_eps = min_error[0][1]