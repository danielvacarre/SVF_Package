from pandas import read_csv

from SVF import CrossValidation

if __name__ == '__main__':
    ruta_datos = "../data/datos.csv"

    inputs = ["x1", "x2"]
    outputs = ["y1", "y2"]

    C = [1]
    eps = [0]
    d = [2]

    data_simulation = read_csv(ruta_datos, sep=";")

    cv = CrossValidation("SVF-SP", inputs, outputs, data_simulation, C, eps, d)
