from pandas import read_csv

from SVF_Methods import CrossValidation, SVF, SVF_GRID

if __name__ == '__main__':
    ruta_datos = "../data/datos.csv"

    inputs = ["x1","x2","y1", "y2"]
    outputs = ["y1", "y2"]
    method = "SSVF"

    C = [1]
    eps = [0,1]
    d = [2]

    data_simulation = read_csv(ruta_datos, sep=";")
    #
    # cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d,n_folds=2)
    #
    # cross_validation.cv()
    # #
    # # model = SVF(method, inputs, outputs, data_simulation, 1, 0, 2)
    # # model.train()

    grid = SVF_GRID(data_simulation, inputs, 2)
    grid.create_grid()
    # position = grid.search_observation([2.5,4])
    # print(grid.calculate_phi_observation(position))