from pandas import read_csv

from svf_package.grid.svf_grid import SVF_GRID
from svf_package.grid.svf_splines_grid import SVF_SPLINES_GRID
from svf_package.svf_functions import train

if __name__ == '__main__':
    ruta_datos = "./data/datos.csv"
    inputs = ["x1"]
    outputs = ["y1"]
    method = "SVF-SP"

    C = [1, 2, 5, 10, 100]
    eps = [0, 1]
    d = [2, 4, 6]
    "./data/datos.csv"
    data_simulation = read_csv(ruta_datos, sep=";")

    # cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, n_folds=2)
    # cross_validation.cv()

    svfc = train("SVF", inputs, outputs, data_simulation, 1, 0, 2)
    svfc.solve()
    prediction_svfc = svfc.estimation([1])
    ssvf = train("SSVF", inputs, outputs, data_simulation, 1, 0, 2)
    ssvf.solve()
    prediction_ssvf = ssvf.estimation([1])
    svf_sp = train("SVF-SP", inputs, outputs, data_simulation, 1, 0, 2)
    svf_sp.solve()
    prediction_svf_sp = svf_sp.estimation([1])

