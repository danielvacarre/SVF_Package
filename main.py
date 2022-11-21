from pandas import read_csv

from svf_package.cv.cv import CrossValidation
from svf_package.grid.svf_splines_grid import SVF_SPLINES_GRID
from svf_package.svf import SVF
from svf_package.svf_functions import train, modify_model

if __name__ == '__main__':
    ruta_datos = "./data/datos.csv"

    inputs = ["x1","x2"]
    outputs = ["y1", "y2"]
    method = "SVF-SP"

    C = [1, 2, 5, 10, 100]
    eps = [0, 1]
    d = [2, 4, 6]

    data_simulation = read_csv(ruta_datos, sep=";")

    # cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, n_folds=2)
    #
    # cross_validation.cv()
    #
    # cross2 = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, ts=0.5)
    #
    # cross2.cv()

    svf_splines = train("SVF-SP", inputs, outputs, data_simulation, 1, 0, 2)
    ssvf = train("SSVF", inputs, outputs, data_simulation, 1, 0, 2)
    svf_splines.model = modify_model(svf_splines, 1000, 1)
    ssvf.model = modify_model(ssvf, 1000, 1)
    # grid = SVF_SPLINES_GRID(data_simulation, inputs, 2)
    #
    # grid.create_grid()