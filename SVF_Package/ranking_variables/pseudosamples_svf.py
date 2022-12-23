from numpy import median
from pandas import DataFrame, concat

from svf_package.cv.cv import CrossValidation
from svf_package.ranking_variables.pseudosample import Pseudosample
from svf_package.ranking_variables.ranking_method import RankingMethod
from svf_package.svf_functions import create_SVF, calculate_mse


class Pseudosamples(RankingMethod):

    def __init__(self, svf_method, inputs, outputs, data, C, eps, D, q, verbose=False, n_folds=1, seed=0, stop_criteria=2):

        super().__init__(svf_method, inputs, outputs, data, C, eps, D, verbose, n_folds,
                                    seed, stop_criteria)
        self.q = q
        self.mad_list = None
        self.data_drop = data
        self.pseudo_samples = None
        self.ranking = None
        self.verbose = verbose

    def rank(self):
        num_inp = len(self.inputs)
        self.data_drop = self.data.copy()
        self.pseudo_samples = list()
        self.ranking = list()
        self.mad_list = list()
        cont = 0
        while num_inp >= self.stop_criteria:
            mad_list = list()
            pseudosamples = list()
            self.inputs = self.data_drop[self.data_drop.columns.difference(self.outputs)].columns
            print(self.seed)
            self.cv =  CrossValidation(self.svf_method, self.inputs, self.outputs, self.data_drop, self.C, self.eps, self.D, self.seed, self.n_folds,self.verbose)
            self.cv.cv()
            svf_obj = create_SVF(self.svf_method, self.inputs, self.outputs, self.data_drop, self.cv.best_C, self.cv.best_eps, self.cv.best_d)
            svf_obj.train()
            svf_obj.solve()
            for inp in range(num_inp):
                pseudo_sample = Pseudosample(self.data_drop, self.inputs,self.outputs,self.q, inp, num_inp)
                pseudo_sample.create_pseudo_sample(num_inp)
                print(pseudo_sample.data)
                pseudo_sample.df_pseudosample_pred = svf_obj.get_new_dataset_estimation(pseudo_sample.data)
                y_pred = pseudo_sample.df_pseudosample_pred.filter(self.outputs)
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
        self.mad_list = DataFrame(self.mad_list)
        ranking_list = list(reversed(self.ranking))
        self.ranking = DataFrame(list(range(1,len(ranking_list)+1)),columns=["RANKING"])
        self.ranking = concat([self.ranking,DataFrame(ranking_list,columns=["VAR"])],axis=1)

    def drop_variable(self, cont):
        mad = 1e10
        drop_var = None
        for pseud in self.pseudo_samples[cont]:
            if pseud.mad_p < mad:
                drop_var = pseud.col_name
                mad = pseud.mad_p
        self.data_drop = self.data_drop = self.data_drop.drop(drop_var, axis=1)
        self.ranking.append(drop_var)