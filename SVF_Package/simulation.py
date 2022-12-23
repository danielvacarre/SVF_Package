from os import path, makedirs

from numpy import mean
from pandas import DataFrame


class Simulation(object):
    """ Simulation
    """

    def __init__(self):
        self.scenario = None
        self.size = None
        self.nombre = None
        self.time_df = None
        self.time_df_agg = None
        self.hiper_list = list()
        self.mse_list = list()
        self.bias_list = list()
        self.time_list = list()
        self.hiper_df = None
        self.hiper_df_agg = None
        self.mse_df = None
        self.hiper_df_agg = None
        self.bias_df = None
        self.bias_df_agg = None
        self.cv = None
        self.svf_obj = None

    def value_true_frontier(self, scenario, x_df):
        if scenario == "B1":
            value = 3 + x_df.x1 ** 0.5

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

    def true_fontier(self, scenario, data, inputs):
        x_df = data.filter(inputs)
        y_list = []
        for i in range(0, len(x_df)):
            # print(i)
            x = x_df.loc[i]
            df_est = self.value_true_frontier(scenario, x)
            y_list.append(df_est)
        return y_list

    def get_mse_metrics(self, file, estimation):
        split = file.split('_')
        num = split[0]
        scenario = split[1]
        size = split[2].split('.')[0]
        mse_dea = round(mean((estimation.y_fron - estimation.y_DEA) ** 2), 5)
        mse_csvf = round(mean((estimation.y_fron - estimation.y_CSVF) ** 2), 5)
        error = {'Num': num,
                 'SCENARIO': scenario,
                 'SIZE': size,
                 'DEA': mse_dea,
                 'CSVF': mse_csvf,
                 'FRACTION OF TRIALS': compare(mse_csvf, mse_dea),
                 'IMPROVEMENTS': round(mean((mse_dea - mse_csvf) / mse_dea * 100), 3)
                 }
        return error

    def get_bias_metrics(self, file, estimation):
        split = file.split('_')
        num = split[0]
        scenario = split[1]
        size = split[2].split('.')[0]
        bias_dea = round(mean(estimation.y_fron - estimation.y_DEA), 5)
        bias_csvf = round(mean(estimation.y_fron - estimation.y_CSVF), 5)
        bias_dea_abs = round(mean(abs(estimation.y_fron - estimation.y_DEA)), 5)
        bias_csvf_abs = round(mean(abs(estimation.y_fron - estimation.y_CSVF)), 5)
        error = {'Num': num,
                 'SCENARIO': scenario,
                 'SIZE': size,
                 'DEA': bias_dea,
                 'CSVF': bias_csvf,
                 'FRACTION OF TRIALS': compare(bias_csvf, bias_dea),
                 'IMPROVEMENTS': round(mean((bias_dea - bias_csvf) / bias_dea * 100), 3),
                 '|DEA|': bias_dea_abs,
                 '|CSVF|': bias_csvf_abs,
                 '|FRACTION OF TRIALS|': compare(bias_csvf_abs, bias_dea_abs),
                 '|IMPROVEMENTS|': round(mean((bias_dea_abs - bias_csvf_abs) / bias_dea_abs * 100), 3)
                 }
        return error

    def get_time_metrics(self, file):
        split = file.split('_')
        num = split[0]
        scenario = split[1]
        size = split[2].split('.')[0]
        time = {'Num': num,
                'SCENARIO': scenario,
                'SIZE': size,
                'CV': self.cv.cv_time.total_seconds(),
                'BM': self.svf_obj.train_time.total_seconds(),
                'SBM': self.svf_obj.solve_time.total_seconds()
                 }
        return time

    def get_hip_metrics(self, file, best_c, best_eps, best_d):
        split = file.split('_')
        num = split[0]
        scenario = split[1]
        size = split[2].split('.')[0]
        error = {'Num': num,
                 'SCENARIO': scenario,
                 'SIZE': size,
                 'C': best_c,
                 'EPS': best_eps,
                 'D': best_d}
        return error

    def get_agg_hip_metrics(self):

        mean_error_c_min = self.hiper_df['C'].min()
        mean_error_c_max = self.hiper_df['C'].max()
        mean_error_eps_min = self.hiper_df['EPS'].min()
        mean_error_eps_max = self.hiper_df['EPS'].max()
        mean_error_d_min = self.hiper_df['D'].min()
        mean_error_d_max = self.hiper_df['D'].max()

        mean_error = self.hiper_df.groupby(['SCENARIO', 'SIZE']).mean()
        std_error = self.hiper_df.groupby(['SCENARIO', 'SIZE']).std()

        self.hiper_df_agg = DataFrame()
        self.hiper_df_agg['SCENARIO'] = self.hiper_df.SCENARIO
        self.hiper_df_agg['SIZE'] = self.hiper_df.SIZE
        self.hiper_df_agg['C'] = str(round(mean_error["C"].values[0], 3)) + ' (' + str(
            round(std_error["C"].values[0], 3)) + ')'
        self.hiper_df_agg['MIN_C'] = mean_error_c_min
        self.hiper_df_agg['MAX_C'] = mean_error_c_max
        self.hiper_df_agg['EPS'] = str(round(mean_error["EPS"].values[0], 3)) + ' (' + str(
            round(std_error["EPS"].values[0], 3)) + ')'
        self.hiper_df_agg['MIN_EPS'] = mean_error_eps_min
        self.hiper_df_agg['MAX_EPS'] = mean_error_eps_max
        self.hiper_df_agg['D'] = str(round(mean_error["D"].values[0], 3)) + ' (' + str(
            round(std_error["D"].values[0], 3)) + ')'
        self.hiper_df_agg['MIN_D'] = mean_error_d_min
        self.hiper_df_agg['MAX_D'] = mean_error_d_max
        self.hiper_df_agg = self.hiper_df_agg.drop_duplicates()

    def get_agg_mse_metrics(self):

        mean_error = self.mse_df.groupby(['SCENARIO', 'SIZE']).mean()
        std_error = self.mse_df.groupby(['SCENARIO', 'SIZE']).std()

        self.mse_df_agg = DataFrame()
        self.mse_df_agg['SCENARIO'] = self.hiper_df.SCENARIO
        self.mse_df_agg['SIZE'] = self.hiper_df.SIZE
        self.mse_df_agg['DEA'] = str(round(mean_error["DEA"].values[0], 3)) + ' (' + \
                                 str(round(std_error["DEA"].values[0], 3)) + ')'
        self.mse_df_agg['CSVF'] = str(round(mean_error["CSVF"].values[0], 3)) + ' (' + \
                                  str(round(std_error["CSVF"].values[0], 3)) + ')'
        self.mse_df_agg['FRACTION OF TRIALS'] = round(mean_error["FRACTION OF TRIALS"].values[0], 3)
        self.mse_df_agg['IMPROVEMENTS'] = str(round(mean_error["IMPROVEMENTS"].values[0], 3)) + ' (' + \
                                          str(round(std_error["IMPROVEMENTS"].values[0], 3)) + ')'
        self.mse_df_agg = self.mse_df_agg.drop_duplicates()

    def get_agg_bias_metrics(self):
        # print(self.bias_df)
        mean_error = self.bias_df.groupby(['SCENARIO', 'SIZE']).mean()
        std_error = self.bias_df.groupby(['SCENARIO', 'SIZE']).std()
        # print(mean_error)
        self.bias_df_agg = DataFrame()
        self.bias_df_agg['SCENARIO'] = self.hiper_df.SCENARIO
        self.bias_df_agg['SIZE'] = self.hiper_df.SIZE
        self.bias_df_agg['DEA'] = str(round(mean_error["DEA"].values[0], 3)) + ' (' + str(
            round(std_error["DEA"].values[0], 3)) + ')'
        self.bias_df_agg['CSVF'] = str(round(mean_error["DEA"].values[0], 3)) + ' (' + str(
            round(std_error["CSVF"].values[0], 3)) + ')'
        self.bias_df_agg['FRACTION OF TRIALS'] = round(mean_error["FRACTION OF TRIALS"].values[0], 3)
        self.bias_df_agg['IMPROVEMENTS'] = str(round(mean_error["IMPROVEMENTS"].values[0], 3)) + ' (' + str(
            round(std_error["IMPROVEMENTS"].values[0], 3)) + ')'
        self.bias_df_agg['|DEA|'] = str(round(mean_error["|DEA|"].values[0], 3)) + ' (' + str(
            round(std_error["|DEA|"].values[0], 3)) + ')'
        self.bias_df_agg['|CSVF|'] = str(round(mean_error["|CSVF|"].values[0], 3)) + ' (' + str(
            round(std_error["|CSVF|"].values[0], 3)) + ')'
        self.bias_df_agg['|FRACTION OF TRIALS|'] = round(mean_error["|FRACTION OF TRIALS|"].values[0], 3)
        self.bias_df_agg['|IMPROVEMENTS|'] = str(round(mean_error["|IMPROVEMENTS|"].values[0], 3)) + ' (' + str(
            round(std_error["|IMPROVEMENTS|"].values[0], 3)) + ')'
        self.bias_df_agg = self.bias_df_agg.drop_duplicates()

    def get_agg_time_metrics(self):
        # print(self.bias_df)
        mean_error = self.time_df.groupby(['SCENARIO', 'SIZE']).mean()
        std_error = self.time_df.groupby(['SCENARIO', 'SIZE']).std()

        self.time_df_agg = DataFrame()
        self.time_df_agg['SCENARIO'] = self.hiper_df.SCENARIO
        self.time_df_agg['SIZE'] = self.hiper_df.SIZE
        self.time_df_agg['CV'] = str(round(mean_error["CV"].values[0], 3)) + ' (' + str(
            round(std_error["CV"].values[0], 3)) + ')'
        self.time_df_agg['BM'] = str(round(mean_error["BM"].values[0], 3)) + ' (' + str(
            round(std_error["BM"].values[0], 3)) + ')'
        self.time_df_agg['SBM'] = str(round(mean_error["SBM"].values[0], 3)) + ' (' + str(
            round(std_error["SBM"].values[0], 3)) + ')'
        self.time_df_agg = self.time_df_agg.drop_duplicates()

    def guardar_registro_fichero(self, df, name_dir):
        dir = "./logs_"+ self.cv.method +"/" + name_dir + "/"
        if not path.exists(dir):
            makedirs(dir)
        name_file = name_dir + "_" + self.nombre
        name_directory = dir + name_file + ".csv"
        df.to_csv(name_directory, sep=";", index=True, decimal=",")

    def guardar_resultados_fichero(self,df, name_dir):
        dir = "./resultados_" + self.cv.method + "/" + name_dir + "/"
        if not path.exists(dir):
            makedirs(dir)
        name_file = dir + name_dir + "_" + str(self.scenario) + "_" + str(self.size) + ".csv"
        df.to_csv(name_file, sep=";", header=True, index=False, decimal=",")

def compare(a, b):
    if a <= b:
        return 1
    else:
        return 0
