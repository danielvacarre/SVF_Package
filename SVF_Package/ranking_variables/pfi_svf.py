from numpy.random import permutation
from pandas import DataFrame, concat

from svf_package.cv.cv import CrossValidation
from svf_package.ranking_variables.ranking_method import RankingMethod
from svf_package.svf_functions import create_SVF, calculate_mse


class PFI(RankingMethod):

    def __init__(self, svf_method, inputs, outputs, data, C, eps, D, verbose, n_folds, seed, stop_criteria):

        super().__init__(svf_method, inputs, outputs, data, C, eps, D, verbose, n_folds,
                         seed, stop_criteria)

    def rank(self):
        n_inp = len(self.inputs)
        self.cv_obj = CrossValidation(self.svf_method, self.inputs, self.outputs, self.data, self.C, self.eps, self.D,
                                  self.seed, self.verbose, self.n_folds)
        self.cv_obj.cv()
        data_train = self.cv_obj.folds[0].data_train
        data_test = self.cv_obj.folds[0].data_test
        self.error_original = self.cv_obj.results['mse'].min()
        self.error_inputs = list()
        self.ranking = DataFrame(columns=['var', 'score'])
        for inp in range(n_inp):
            if self.verbose == True:
                print("Evaluating ranking of variable " + self.inputs[inp])
            data_train.iloc[:, inp] = permutation(data_train.iloc[:, inp].values)
            results_by_fold = DataFrame(columns=["C", "eps", "d"])
            for d in self.D:
                svf_obj = create_SVF(self.svf_method, self.inputs, self.outputs, data_train, 1, 0, d)
                svf_obj.train()
                for c in self.C:
                    for e in self.eps:
                        if self.verbose == True:
                            print("     C:", c, "EPS:", e, "D:", d)
                        svf_obj.model = svf_obj.modify_model(c, e)
                        svf_obj.solve()
                        mse = calculate_mse(svf_obj, data_test)
                        results_by_fold = concat([results_by_fold,
                                                  DataFrame.from_records([{'Num': "RANKING" + str(inp + 1),
                                                                           'C': c,
                                                                           "eps": e,
                                                                           "d": d,
                                                                           "mse": mse}])
                                                  ])
            error_input = results_by_fold['mse'].min()
            self.error_inputs.append(error_input)
            # El score se calcula como el error del mejor modelo de ese input
            # menos el error del conjunto de datos original entre el error original por 100
            score = (error_input - self.error_original) / self.error_original * 100
            self.ranking = self.ranking.append(
                {
                    "var": self.inputs[inp],
                    "score": score
                },
                ignore_index=True,
            )
        # Quien tenga mayor score es el que mayor importancia tiene la variable
        self.ranking = self.ranking.sort_values(by=['score'], ascending=False).reset_index(drop=True)
