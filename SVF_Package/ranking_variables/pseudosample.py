from numpy import quantile, zeros
from pandas import DataFrame, concat

class Pseudosample:

    def __init__(self, data, inputs, outputs, q, inp, max_num_inp):
        self.q = q
        self.outputs = outputs
        self.inputs = inputs
        self.inp = inp
        self.max_num_inp = max_num_inp
        self.data = data
        self.df_pseudosample_pred = None
        self.mad_p = 0
        self.col_name = None

    def create_pseudo_sample(self, num_inp):
        """
        :int num_inp:
        :return:
        """
        self.col_name = self.inputs[self.inp]
        name_columns = self.data.filter(self.inputs).columns
        df_pseudo = DataFrame(zeros((self.q, num_inp)), columns=name_columns)
        for row in range(self.q):
            for col in range(num_inp):
                if col == self.inp:
                    df_pseudo.iloc[row, col] = quantile(self.data.iloc[:, col], 1 / self.q * row)
                else:
                    df_pseudo.iloc[row, col] = self.data.iloc[:, col].mean()
        Y = self.data.filter(self.outputs)
        Y = Y.iloc[0:self.q]
        df_pseudo = concat((df_pseudo, Y), axis=1)
        self.data = df_pseudo


