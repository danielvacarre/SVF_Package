from matplotlib.pyplot import plot, show
from numpy import unique
from pandas import read_csv

from svf_package.svf_functions import create_SVF


def calculate_d(n_obs):
    d = list()
    for i in range(1, 11):
        n = int(round(0.1 * i * n_obs, 0))
        if n > 0:
            d.append(n)
    d = unique(d).tolist()
    return d

if __name__ == '__main__':

    # inputs = [["x1"],["x1","x2"],["x1","x2","y1"]]
    # outputs = [["y1"]]
    # prediccion = [[3],[1,2],[1,3,1]]
    # c = 1
    # eps = 0
    # d = 2
    # method_list = ["SVF","SSVF","dual","SVF-SP"]
    # for i in inputs:
    #     cont = len(i)-1
    #     for o in outputs:
    #         for method in method_list:
    #             print("N_INP:", i," N_OUT:", o)
    #             svf = create_SVF(method, i, o, data, c, eps, d)
    #             svf.train()
    #             # print(svf.model.export_to_string())
    #             svf.solve()
    #             print(method, svf.solution.w, "=>", svf.get_estimation(prediccion[cont]))
    #             print("***********************************************************")
    # print("=======================================================================")
    data_simulation = read_csv("C:/Users/Master/Desktop/SVF_Package/data/data_3_100/18_3_100.csv",sep=";")
    method = "SVF"
    inputs = ["x1","x2","x3"]
    outputs = ["y"]
    svf_obj = create_SVF(method, inputs, outputs, data_simulation, 100, 0.05, 20)
    svf_obj.train()
    svf_obj.solve()
    # svf_obj2 = create_SVF(method, inputs, outputs, data_simulation, 100, 0, 50)
    # svf_obj2.train()
    # svf_obj2.solve()

    # df = svf_obj.get_df_all_estimation()
    # df2 = svf_obj2.get_df_all_estimation()
    #
    # plot(df.x1, df.y_CSVF,"red")
    # plot(df.x1, df.y_DEA,"black")
    # plot(df.x1, df2.y_CSVF,"blue")
    # show()
