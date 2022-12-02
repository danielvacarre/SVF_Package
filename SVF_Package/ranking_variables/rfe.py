from numpy import array, dot
from pandas import DataFrame

from svf_package.cv.cv import CrossValidation
from svf_package.methods.svf_dual import SVFDual


class RFESVF:
    def __init__(self, svf_method, inputs, outputs, data, C, eps, D, n_folds, seed, stop_criteria):
        self.seed = seed
        self.n_folds = n_folds
        self.D = D
        self.eps = eps
        self.C = C
        self.data = data
        self.outputs = outputs
        self.inputs = inputs
        self.svf_method = svf_method
        self.dj_l = None
        self.stop_criteria = stop_criteria
        self.list_problem = None
        self.ranking = None
        self.cv_obj = None
    
    def rank(self):
        n_dim = len(self.inputs)
        list_problems = list()
        list_dj = list()
        list_ranking = list()
        while n_dim >= self.stop_criteria:
            self.cv_obj = CrossValidation(self.svf_method, self.inputs, self.outputs, self.data, self.C, self.eps, self.D,verbose=True)
            self.cv_obj.cv()
            svf_dual = SVFDual("dual", self.inputs, self.outputs, self.data, self.cv_obj.best_C, self.cv_obj.best_eps, self.cv_obj.best_d)
            svf_dual.train()
            svf_dual.solve()
            print(svf_dual.solution.w)
            n_dim -= 1
    #TODO: Hay que comprobar si esto funciona con los cambios hechos en el algoritmo
    def calculate_dj_l(self, svf_model):
        dj_l = DataFrame(columns=["var", "dj"])
        for l in range(len(svf_model.inputs)):
            matrix_phi_l = svf_model.calculate_matrix_transformations_without_l(l)
            s1 = 0
            s2 = 0
            for r in range(len(svf_model.y_cols)):
                for i in range(len(svf_model.data)):
                    s2 += (svf_model.solution.alpha[r][i] - svf_model.solution.delta[r][i]) * \
                          (array(svf_model.matrix_phi[i]) - array(matrix_phi_l[i]), svf_model.solution.gamma[r])
                    for j in range(len(svf_model.data)):
                        s1 += (svf_model.solution.alpha[r][i] - svf_model.solution.delta[r][i]) * \
                              (svf_model.solution.alpha[r][j] - svf_model.solution.delta[r][j]) * \
                              (dot(svf_model.matrix_phi[i], svf_model.matrix_phi[j]) -
                               dot(matrix_phi_l[i], matrix_phi_l[j])
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