from pandas import read_csv

from svf_package.cv.cv import CrossValidation

if __name__ == '__main__':
    ruta_datos = "./data/datos.csv"

    inputs = ["x1","x2"]
    outputs = ["y1", "y2"]
    method = "SSVF"

    C = [1,2,5,10,100]
    eps = [0,1]
    d = [2,4,6]

    data_simulation = read_csv(ruta_datos, sep=";")

    cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, n_folds=2)

    cross_validation.cv()

    cross2 = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, ts=0.5)

    cross2.cv()
