from pandas import read_csv

from svf_package.cv.cv import CrossValidation

if __name__ == '__main__':
    ruta_datos = "./data/2_3_30.csv"
    inputs = ["x1","x2","x3"]
    outputs = ["y"]
    method = "SVF"

    C = [1e-3,1e-2,1e-1,1,10,100,1000]
    eps = [0, 1e-3, 1e-2, 1e-1, 1]
    d = [3,6,9,12]

    data_simulation = read_csv(ruta_datos, sep=";")

    method_list = ["SVF-SP","SVF","SSVF"]
    cross_list = list()
    for method in method_list:
        print(method)
        cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, verbose=True)
        cross_validation.cv()
        cross_list.append(cross_validation)
