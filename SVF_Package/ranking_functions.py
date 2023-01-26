from svf_package.ranking_variables.svf_pfi import PFI
from svf_package.ranking_variables.svf_pseudosamples import SVFPseudosamples
from svf_package.ranking_variables.svf_rfe import RFESVF




def create_ranking(method_ranking, svf_method, inputs, outputs, data, c, eps, d, verbose=False, n_folds=1, seed=0, stop_criteria=2):
    #TODO: Pseudosamples
    if method_ranking == "RFE":
        ranking_var = RFESVF(svf_method, inputs, outputs, data, c, eps, d, verbose, n_folds, seed, stop_criteria)
    elif method_ranking == "PFI":
        ranking_var = PFI(svf_method, inputs, outputs, data, c, eps, d, verbose, n_folds, seed, stop_criteria)
    elif method_ranking == "Pseudosamples":
        ranking_var = SVFPseudosamples(svf_method, inputs, outputs, data, c, eps, d, verbose, n_folds, seed, stop_criteria)
    else:
        raise RuntimeError("The method selected doesn't exist")
    return ranking_var