from svf_package.ranking_variables.pfi_svf import PFI
from svf_package.ranking_variables.rfe_svf import RFESVF


def create_ranking(method_ranking, svf_method, inputs, outputs, data, c, eps, d, verbose=False, n_folds=1, seed=0, stop_criteria=2):
    #TODO:
    if method_ranking == "RFE":
        ranking_var = RFESVF(svf_method, inputs, outputs, data, c, eps, d, verbose, n_folds, seed, stop_criteria)
    elif method_ranking == "PFI":
        ranking_var = PFI(svf_method, inputs, outputs, data, c, eps, d, verbose, n_folds, seed, stop_criteria)
    elif method_ranking == "Pseudosamples":
        pass
        # svf = SVFC(method_svf, inputs, outputs, data, c, eps, d)
    elif method_ranking == "Bootstrap":
        pass
        # svf = SVFC(method_svf, inputs, outputs, data, c, eps, d)
    else:
        raise RuntimeError("The method selected doesn't exist")
    return ranking_var