from pandas import read_csv

from SVF_Methods import CrossValidation, SVF

if __name__ == '__main__':
    ruta_datos = "../data/datos.csv"

    inputs = ["x1", "x2"]
    outputs = ["y1", "y2"]
    method = "SVF-SP"

    C = [1]
    eps = [0]
    d = [2]

    data_simulation = read_csv(ruta_datos, sep=";")

    cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d,n_folds=2)

    cross_validation.cv()
