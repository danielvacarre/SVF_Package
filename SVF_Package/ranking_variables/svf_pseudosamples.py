from numpy import median
from pandas import DataFrame

from svf_package.cv.cv import CrossValidation
from svf_package.ranking_variables.pseudosample import Pseudosample
from svf_package.ranking_variables.ranking_method import RankingMethod

#TODO: Comentar
from svf_package.svf_functions import create_SVF


class SVFPseudosamples(RankingMethod):

    def __init__(self, svf_method, inputs, outputs, data, C, eps, D, verbose, n_folds=1, seed=0, stop_criteria=2):

        super().__init__(svf_method, inputs, outputs, data, C, eps, D, verbose, n_folds,
                         seed, stop_criteria)

        self.mad_list = None
        self.data_drop = data.copy()
        self.pseudo_samples = None
        self.ranking = None

    def rank(self):
        num_inp = len(self.inputs)
        self.pseudo_samples = list()
        self.ranking = list()
        self.mad_list = list()
        cont = 0
        while num_inp >= self.stop_criteria:
            print(num_inp, ">=", self.stop_criteria, self.inputs)
            mad_list = list()
            pseudosamples = list()
            self.cv_obj = CrossValidation(self.svf_method, self.inputs, self.outputs, self.data_drop, self.C, self.eps, self.D,
                                      self.verbose, self.seed, self.n_folds)
            self.cv_obj.cv()
            svf_obj = create_SVF(self.svf_method, self.inputs, self.outputs, self.data,
                                 self.cv_obj.best_C, self.cv_obj.best_eps, self.cv_obj.best_d)
            svf_obj.train()
            svf_obj.solve()
            for inp in range(num_inp):
                pseudo_sample = Pseudosample(self.data_drop, self.inputs,self.outputs, 10, inp, num_inp)
                pseudo_sample.create_pseudo_sample()
                print(pseudo_sample.data)
                # con las soluciones del problema general, calcular la prediccion y* de la pseudomuestra
                svf_obj.data = pseudo_sample.data
                svf_obj.get_df_estimation()
                pseudo_sample.df_pseudosample_pred = svf_obj.df_estimation
                y_pred = pseudo_sample.df_pseudosample_pred.filter(self.outputs)
                # calculamos MAD: con el vector de predicciones de la pseudomuestra
                mad_p = median(abs(y_pred-median(y_pred)))
                pseudo_sample.mad = mad_p
                mad_list.append(mad_p)
                pseudosamples.append(pseudo_sample)
            self.pseudo_samples.append(pseudosamples)
            self.mad_list.append(mad_list)
            self.drop_variable(cont)
            num_inp -= 1
            cont += 1
        cols = self.data_drop[self.data_drop.columns.difference(self.outputs)].columns
        for col in cols:
            self.ranking.append(col)
        self.ranking = list(reversed(self.ranking))
        self.mad_list = DataFrame(self.mad_list)
        self.ranking = DataFrame(self.ranking)

    def  drop_variable(self, cont):
        mad = 1e10
        drop_var = None
        for pseud in self.pseudo_samples[cont]:
            if pseud.mad_p < mad:
                drop_var = pseud.col_name
                mad = pseud.mad_p
        self.data_drop = self.data_drop = self.data_drop.drop(drop_var, axis=1)
        self.inputs.remove(drop_var)
        self.ranking.append(drop_var)