from pandas import read_csv

from svf_package.grid.svf_splines_grid import SVF_SPLINES_GRID
from svf_package.svf_functions import train

if __name__ == '__main__':
    ruta_datos = "./data/datos.csv"
    inputs = ["x1"]
    outputs = ["y1","y2"]
    method = "SVF-SP"

    C = [1, 2, 5, 10, 100]
    eps = [0, 1]
    d = [2, 4, 6]

    data_simulation = read_csv(ruta_datos, sep=";")

    # cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, n_folds=2)
    # cross_validation.cv()

    svf_splines = SVF_SPLINES_GRID(data_simulation, inputs, 2)
    svf_splines.create_grid()

    svfc = train("SVF", inputs, outputs, data_simulation, 100, 0, 2)
    svfc.solve()
    ssvf = train("SSVF", inputs, outputs, data_simulation, 100, 0, 2)
    ssvf.solve()
    svf_sp = train("SVF-SP", inputs, outputs, data_simulation, 100, 0, 2)
    svf_sp.solve()
    prediction = svf_sp.estimation([3])
    prediction2 = svfc.estimation([3])
    prediction3 = ssvf.estimation([3])