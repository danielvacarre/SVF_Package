from numpy import mean
from pandas import read_csv, concat, merge, DataFrame
from svf_package.cv.cv import CrossValidation
from svf_package.metrics import Metrics
from svf_package.svf_functions import create_SVF


def value_true_frontier(scenario, x_df):
    if scenario == "1":
        value = x_df.x1 ** 0.5

    if scenario == "2":
        value = x_df.x1 ** 0.4 * \
                x_df.x2 ** 0.1

    if scenario == "3":
        value = x_df.x1 ** 0.3 * \
                x_df.x2 ** 0.1 * \
                x_df.x3 ** 0.1

    if scenario == "4":
        value = x_df.x1 ** 0.3 * \
                x_df.x2 ** 0.1 * \
                x_df.x3 ** 0.08 * \
                x_df.x4 ** 0.02

    if scenario == "5":
        value = x_df.x1 ** 0.3 * \
                x_df.x2 ** 0.1 * \
                x_df.x3 ** 0.08 * \
                x_df.x4 ** 0.02 * \
                x_df.x5 ** 0.02

    if scenario == "6":
        value = x_df.x1 ** 0.3 * x_df.x2 ** 0.1 * x_df.x3 ** 0.08 * \
                x_df.x4 ** 0.01 * x_df.x5 ** 0.006 * x_df.x6 ** 0.004

    if scenario == "9":
        value = x_df.x1 ** 0.3 * x_df.x2 ** 0.1 * x_df.x3 ** 0.08 * \
                x_df.x4 ** 0.005 * x_df.x5 ** 0.004 * x_df.x6 ** 0.001 * \
                x_df.x7 ** 0.005 * x_df.x8 ** 0.004 * x_df.x9 ** 0.001

    if scenario == "12":
        value = x_df.x1 ** 0.2 * x_df.x2 ** 0.075 * x_df.x3 ** 0.025 * \
                x_df.x4 ** 0.05 * x_df.x5 ** 0.05 * x_df.x6 ** 0.08 * \
                x_df.x7 ** 0.005 * x_df.x8 ** 0.004 * x_df.x9 ** 0.001 * \
                x_df.x10 ** 0.005 * x_df.x11 ** 0.004 * x_df.x12 ** 0.001

    if scenario == "15":
        value = x_df.x1 ** 0.15 * x_df.x2 ** 0.025 * x_df.x3 ** 0.025 * \
                x_df.x4 ** 0.05 * x_df.x5 ** 0.025 * x_df.x6 ** 0.025 * \
                x_df.x7 ** 0.05 * x_df.x8 ** 0.05 * x_df.x9 ** 0.08 * \
                x_df.x10 ** 0.005 * x_df.x11 ** 0.004 * x_df.x12 ** 0.001 * \
                x_df.x13 ** 0.005 * x_df.x14 ** 0.004 * x_df.x15 ** 0.001


    return value

def true_fontier(scenario,data,inputs):
    x_df = data.filter(inputs)
    y_list = []
    for i in range(0, len(x_df)):
        # print(i)
        x = x_df.loc[i]
        df_est = value_true_frontier(scenario, x)
        y_list.append(df_est)
    return y_list

def get_rmse_bias(estimation, f, best_c, best_eps, best_d):
    split = f.split('_')
    num = split[0]
    scenario = split[1]
    size = split[2].split('.')[0]
    mse_dea = round(mean((estimation.y_fron - estimation.y_DEA) ** 2), 5)
    mse_csvf = round(mean((estimation.y_fron - estimation.y_CSVF) ** 2), 5)
    bias_dea = round(mean(estimation.y_fron - estimation.y_DEA), 5)
    bias_csvf = round(mean(estimation.y_fron - estimation.y_CSVF), 5)
    bias_dea_abs = round(mean(abs(estimation.y_fron - estimation.y_DEA)), 5)
    bias_csvf_abs = round(mean(abs(estimation.y_fron - estimation.y_CSVF)), 5)
    error = {'Num': num,
             'Scenario': scenario,
             'Size': size,
             'BEST_C': best_c,
             'BEST_EPS': best_eps,
             'BEST_D': best_d,
             'MSE_DEA': mse_dea,
             'MSE_CSVF': mse_csvf,
             'MSE_CSVF<=MSE_DEA': compare(mse_csvf, mse_dea),
             '%MEJORA_MSE_CSVF': round(mean((mse_dea - mse_csvf) / mse_dea * 100), 3),
             'BIAS_DEA': bias_dea,
             'BIAS_CSVF': bias_csvf,
             'BIAS_DEA_ABS': bias_dea_abs,
             'BIAS_CSVF_ABS': bias_csvf_abs,
             'BIAS_CSVF<=BIAS_DEA': compare(bias_csvf, bias_dea),
             '%MEJORA_BIAS_CSVF': round(mean((bias_dea - bias_csvf) / bias_dea * 100), 3),
             'BIAS_CSVF_ABS<=BIAS_DEA_ABS': compare(bias_csvf_abs, bias_dea_abs),
             '%MEJORA_BIAS_CSVF_ABS': round(mean((bias_dea_abs - bias_csvf_abs) / bias_dea_abs * 100), 3)}
    return error


if __name__ == '__main__':
    ruta_datos = "./data/1_1_30.csv"
    inputs = ["x1"]
    outputs = ["y"]
    method = "SSVF"

    C = [1, 100]
    eps = [0, 3]
    d = [2]

    data_simulation = read_csv(ruta_datos, sep=";")

    cross_validation = CrossValidation(method, inputs, outputs, data_simulation, C, eps, d, verbose=True,n_folds=2)
    cross_validation.cv()

    svf_obj = create_SVF(method, inputs, outputs, data_simulation, 0.5, .1, 6)
    svf_obj.train()
    svf_obj.solve()

    df = svf_obj.get_df_all_estimation()
    df["y_fron"] = true_fontier("1",data_simulation,inputs)


    # metricas = Metrics()
    # metricas.cv_time = cross_validation.cv_time
    # metricas.bm_time = svf_obj.train_time
    # metricas.sbm_time = svf_obj.solve_time



