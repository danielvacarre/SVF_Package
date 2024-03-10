from numpy import dot, array
from pandas import DataFrame, concat
from SVF_Package.cv.cv import CrossValidation
from SVF_Package.ranking_variables.ranking_method import RankingMethod
from SVF_Package.svf_functions import create_SVF


class RFESVF(RankingMethod):

    def __init__(self, svf_method, inputs, outputs, data, C, eps, D, verbose, n_folds=1, seed=0, stop_criteria=2):

        super().__init__(svf_method, inputs, outputs, data, C, eps, D, verbose, n_folds,
                                    seed,stop_criteria)
        self.dj_l = None
        self.list_problem = None
        self.ranking = None
        self.cv_obj = None
    
    def rank(self):
        n_dim = len(self.inputs)
        list_problems = list()
        list_dj = list()
        list_ranking = list()
        while n_dim >= self.stop_criteria:
            if self.verbose == True:
                print("Number of dimension evaluating ", n_dim)
            self.cv_obj = CrossValidation(self.svf_method, self.inputs, self.outputs, self.data, self.C, self.eps, self.D,verbose=True)
            self.cv_obj.cv()
            svf_dual = create_SVF("dual", self.inputs, self.outputs, self.data, self.cv_obj.best_C, self.cv_obj.best_eps, self.cv_obj.best_d)
            svf_dual.train()
            svf_dual.solve()
            dj_df = self.calculate_dj_l(svf_dual)
            list_problems.append(svf_dual)
            list_dj.append(dj_df.values)
            var_remove = dj_df[dj_df.dj == dj_df.dj.min()]
            var_remove = var_remove['var'].values[0]
            list_ranking.append(var_remove)
            self.inputs.remove(var_remove)
            n_dim = len(self.inputs)
        for col in self.inputs:
            list_ranking.append(col)
        self.list_problem = list_problems
        list_ranking = list(reversed(list_ranking))
        self.ranking = DataFrame(list(range(1,len(list_ranking)+1)),columns=["RANKING"])
        self.ranking = concat([self.ranking,DataFrame(list_ranking,columns=["VAR"])],axis=1)
        self.dj_l = DataFrame()
        for element in list_dj:
            self.dj_l = self.dj_l.append(DataFrame(element))

    def calculate_dj_l(self, svf_model):
        dj_l = DataFrame(columns=["var", "dj"])
        for l in range(len(svf_model.inputs)):
            matrix_phi_l = svf_model.calculate_matrix_transformations_without_l(l)
            s1 = 0
            s2 = 0
            for out in range(len(svf_model.outputs)):
                for dmu1 in range(len(svf_model.data)):
                    s2 += (svf_model.solution.alpha[out][dmu1] - svf_model.solution.delta[out][dmu1]) * \
                          dot(array(svf_model.grid.data_grid.phi[dmu1][out]) - array(matrix_phi_l[dmu1]), svf_model.solution.gamma[out])
                    for dmu2 in range(len(svf_model.data)):
                        s1 += (svf_model.solution.alpha[out][dmu1] - svf_model.solution.delta[out][dmu1]) * \
                              (svf_model.solution.alpha[out][dmu2] - svf_model.solution.delta[out][dmu2]) * \
                              (dot(svf_model.grid.data_grid.phi[dmu1][out], svf_model.grid.data_grid.phi[dmu2][out]) -
                               dot(matrix_phi_l[dmu1], matrix_phi_l[dmu2])
                               )
            dj = 1 / 2 * (s1 + 2 * s2)
            dj_l = dj_l.append(
                {
                    "var": svf_model.inputs[l],
                    "dj": dj
                },
                ignore_index=True,
            )
        return dj_l