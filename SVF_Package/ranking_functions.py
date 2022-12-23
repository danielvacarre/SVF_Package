from svf_package.ranking_variables.pfi_svf import PFI
from svf_package.ranking_variables.pseudosamples_svf import Pseudosamples
from svf_package.ranking_variables.rfe_svf import RFESVF


def create_ranking(method_ranking, svf_method, inputs, outputs, data, c, eps, d, verbose, n_folds, seed, stop_criteria):

    if method_ranking == "RFE":
        ranking_var = RFESVF(svf_method, inputs, outputs, data, c, eps, d, verbose, n_folds, seed, stop_criteria)
    elif method_ranking == "PFI":
        ranking_var = PFI(svf_method, inputs, outputs, data, c, eps, d, verbose, n_folds, seed, stop_criteria)
    elif method_ranking == "Pseudosamples":
        ranking_var = Pseudosamples(svf_method, inputs, outputs, data, c, eps, d, 10, verbose, n_folds, seed, stop_criteria)
    else:
        raise RuntimeError("The method selected doesn't exist")
    return ranking_var