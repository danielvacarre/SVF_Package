from datetime import datetime

from pandas import DataFrame, concat
from sklearn.model_selection import KFold, train_test_split

from SVF_Package.cv.fold import FOLD
from SVF_Package.svf_functions import calculate_mse, create_SVF, create_dataset

FMT = "%d-%m-%Y %H:%M:%S"

class CrossValidation(object):
    """ Clase validación cruzada
    """

    def __init__(self, method, inputs, outputs, data, C, eps, D, seed=0, n_folds=0, verbose=False, ts=0.33):
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
            ts (float): indica el porcentaje de datos de test a utilizar en la cv
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
        self.ts = ts
        self.verbose = verbose
        self.results = None
        self.results_by_fold = None
        self.folds = None
        self.best_C = None
        self.best_eps = None
        self.best_d = None
        self.cv_time = None

    def cv(self):
        """Función que ejecuta el tipo de validación cruzada:
                >1: aplica el método k_folds
                
               <=1: aplica el método train-test
        """
        self.data = create_dataset(self.inputs, self.outputs, self.data)
        if self.method == "dual":
            raise "Dual method can not implementate cross validation."
        self.results_by_fold = DataFrame(columns=["C", "eps", "d", "mse"])
        if self.n_folds > 1:
            self.kfolds()
        else:
            self.train_test()

    def kfolds(self):
        """Función que ejecuta la validación cruzada por k-folds
        """
        now = datetime.now()
        fecha_inicio_cv = now.strftime(FMT)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        fold_num = 0
        list_fold = list()
        for train_index, test_index in kf.split(self.data):
            fold_num += 1
            data_train, data_test = self.data.iloc[train_index], self.data.iloc[test_index]
            data_train = data_train.reset_index()
            data_test = data_test.reset_index()
            fold = FOLD(data_train, data_test, fold_num)
            for d in self.D:
                svf_obj = create_SVF(self.method, self.inputs, self.outputs, fold.data_train, 1, 0, d)
                svf_obj.train()
                for c in self.C:
                    for e in self.eps:
                        if self.verbose:
                            print("FOLD:", fold_num, "C:", c, "EPS:", e, "D:", d)
                        svf_obj.model = svf_obj.modify_model(c, e)
                        svf_obj.solve()
                        mse = calculate_mse(svf_obj, fold.data_test)
                        self.results_by_fold = concat([self.results_by_fold ,
                                                       DataFrame.from_records([{ 'FOLD': fold_num,
                                                                                 'C': c,
                                                                                 "eps": e,
                                                                                 "d": d,
                                                                                 "mse": mse}])
                                                       ])
            list_fold.append(fold)
        now = datetime.now()
        fecha_fin_cv = now.strftime(FMT)
        self.cv_time = datetime.strptime(fecha_fin_cv, FMT) - datetime.strptime(fecha_inicio_cv, FMT)
        self.folds = list_fold
        self.results = self.results_by_fold.groupby(['C', 'eps', 'd']).sum() / self.n_folds
        self.results = self.results.sort_values(by=["mse","eps","d"])
        self.results = self.results.drop(['FOLD'], axis=1)
        min_error = self.results.loc[self.results['mse'] == self.results['mse'].min()]
        min_error = DataFrame(list(min_error.index), columns=["C","eps","d"])
        min_error = min_error.loc[min_error['C'] == min_error['C'].max()]
        min_error = min_error.loc[min_error['eps'] == min_error['eps'].min()]
        min_error = min_error.loc[min_error['d'] == min_error['d'].min()]
        self.best_C = min_error["C"].values[0]
        self.best_eps = min_error["eps"].values[0]
        self.best_d = int(min_error["d"].values[0])

    def train_test(self):
        """Función que ejecuta la validación cruzada por un porcentaje de train-test
        """
        now = datetime.now()
        fecha_inicio_cv = now.strftime(FMT)
        data_train, data_test = train_test_split(self.data, test_size=self.ts, random_state=self.seed)
        data_train = data_train.reset_index()
        data_test = data_test.reset_index()
        list_fold = list()
        fold = FOLD(data_train, data_test, "TRAIN-TEST")
        list_fold.append(fold)
        for d in self.D:
            svf_obj = create_SVF(self.method, self.inputs, self.outputs, fold.data_train, 1, 0, d)
            svf_obj.train()
            for c in self.C:
                for e in self.eps:
                    if self.verbose:
                        print("     FOLD:", "TRAIN-TEST", "C:", c, "EPS:", e, "D:", d)
                    svf_obj.model = svf_obj.modify_model(c, e)
                    svf_obj.solve()
                    mse = calculate_mse(svf_obj, fold.data_test)
                    self.results_by_fold = concat([self.results_by_fold,
                                                   DataFrame.from_records([{'FOLD': "TR-TE",
                                                                            'C': c,
                                                                            "eps": e,
                                                                            "d": d,
                                                                            "mse": mse}])
                                                   ])
        now = datetime.now()
        fecha_fin_cv = now.strftime(FMT)
        self.cv_time = datetime.strptime(fecha_fin_cv, FMT) - datetime.strptime(fecha_inicio_cv, FMT)
        self.folds = list_fold
        self.results = self.results_by_fold.sort_index(ascending=False)
        self.results = self.results.drop(['FOLD'], axis=1)
        min_error = self.results[self.results.mse == self.results.mse.min()]
        min_error = min_error.iloc[-1]
        self.best_C = min_error.C
        self.best_eps = min_error.eps
        self.best_d = min_error.d
