from numpy import unique
from pandas import read_csv
from svf_package.ranking_functions import create_ranking

def calculate_d(n_obs):
    d = list()
    for i in range(1, 11):
        n = int(round(0.1 * i * n_obs, 0))
        if n > 0:
            d.append(n)
    d = unique(d).tolist()
    return d

if __name__ == '__main__':

    data = read_csv("data/prueba/3_1_30.csv", sep=";")

    inputs = ["x1","x2","x3"]
    outputs = ["y"]
    svf_method = "SSVF"
    ranking_method = "PFI"
    C = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 2, 5, 10, 100]
    eps = [0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
    C = [1e-3, 1e-2, 1e-1, 1, 1e2, 1e3]
    eps = [0, 1e-3, 1e-2, 1e-1, .2, .5, 1]
    n_obs = 20

    D = calculate_d(n_obs)

    ranking_method = create_ranking(ranking_method, svf_method, inputs, outputs, data, C, eps, D, True, 1, 0, 2)
    ranking_method.rank()

    # ranking_method2 = create_ranking("PFI", svf_method, inputs, outputs, data, C, eps, D, verbose= True)
    # ranking_method2.rank()
    #
