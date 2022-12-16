class RankingMethod:

    def __init__(self, svf_method, inputs, outputs, data, C, eps, D, verbose, n_folds=0, seed=0, stop_criteria=2):
        self.svf_method = svf_method
        self.seed = seed
        self.n_folds = n_folds
        self.D = D
        self.eps = eps
        self.C = C
        self.data = data
        self.outputs = outputs
        self.inputs = inputs
        self.stop_criteria = stop_criteria
        self.cv_obj = None
        self.verbose = verbose