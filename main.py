from pandas import read_csv

from svf_package.cv.cv import CrossValidation
from svf_package.efficiency.dea import DEA
from svf_package.ranking_variables.rfe import RFESVF

if __name__ == '__main__':
    ruta_datos = "./data/2_3_30.csv"
    inputs = ["x1","x2","x3"]
    outputs = ["y"]
    method = "SVF"

    C = [1,2]
    eps = [0,1]
    d = [2,3]

    data_simulation = read_csv(ruta_datos, sep=";")

    # method_svf_list = ["SVF-SP","SVF","SSVF"]
    # cross_list = list()
    # for method in method_svf_list:
    #     print(method)
    #     cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, verbose=True)
    #     cross_validation.cv()
    #     cross_list.append(cross_validation)
    # methods_dea = ["ri", "ro", "ddf", "wa", "rui", "ruo", "erg"]
    # dea_obj = DEA(inputs, outputs, data_simulation, methods_dea)
    # dea_obj.get_efficiencies()
    svf_method = "SSVF"
    rfe = RFESVF(svf_method, inputs, outputs, data_simulation, C, eps, d, 0, 0, 2)
    rfe.rank()