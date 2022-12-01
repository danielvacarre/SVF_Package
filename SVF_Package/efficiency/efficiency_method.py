class EfficiencyMethod:
    def __init__(self, inputs, outputs, data, methods, df_estimation=None):
        self.data = data
        self.df_estimation = df_estimation
        self.methods = methods
        self.outputs = outputs
        self.inputs = inputs
        self.df_eff = None

    def get_efficiencies(self):
        self.df_eff = self.data.copy()
        switch_method = {
            "ri": self.calculate_ri(),
            "ro": self.calculate_ro(),
            "ddf": self.calculate_ddf(),
            "wa": self.calculate_wa(),
            "rui": self.calculate_rui(),
            "ruo": self.calculate_ruo(),
            "erg": self.calculate_erg()
        }
        for met in self.methods:
            self.df_eff[met] = switch_method.get(met)

    def calculate_wa_w_inp(self):
        X = self.data.filter(self.inputs)
        X = X.to_numpy()
        n_inp = len(self.inputs)
        n_out = len(self.outputs)
        w_inp = list()
        for i in range(n_inp):
            ran = (max(X[:, i]) - min(X[:, i]))
            summa = (n_inp + n_out)
            w = 1 / (summa * ran)
            w_inp.append(w)
        return w_inp

    def calculate_wa_w_out(self):
        Y = self.data.filter(self.outputs)
        Y = Y.to_numpy()
        w_out = list()
        n_inp = len(self.inputs)
        n_out = len(self.outputs)
        for i in range(n_out):
            ran = (max(Y[:, i]) - min(Y[:, i]))
            summa = (n_inp + n_out)
            w = 1 / (summa * ran)
            w_out.append(w)
        return w_out
