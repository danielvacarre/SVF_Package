from svf_package.ssvf import SSVF
from svf_package.svf_splines import SVF_SP
from svf_package.svfc import SVFC


# def train(method, inputs, outputs, data, c, eps, d):
#     """ Método para entrenar el modelo generado. Se selecciona el algoritmo seleccionado y se entrena.
#
#     Args:
#         method (string): Método SVF_Methods que se quiere utilizar
#         inputs (list): Inputs a evaluar en el conjunto de dato
#         outputs (list): Outputs a evaluar en el conjunto de datos
#         data (pandas.DataFrame): Conjunto de datos a evaluar
#         c (float): Valor del hiperparámetro C que queremos evaluar
#         eps (float): Valor del hiperparámetro épsilon que queremos evaluar
#         d (int): Valor del hiperparámetro d que queremos evaluar
#
#     Raises:
#         RuntimeError: Indica que el método no existe
#
#     Returns:
#         svf_package.svf.SVF: Devuelve un modelo SVF del método escogido
#     """
#
#     if method == "SVF-SP":
#         svf = SVF_SP(method, inputs, outputs, data, c, eps, d)
#         svf.train()
#     elif method == "SSVF":
#         svf = SSVF(method, inputs, outputs, data, c, eps, d)
#         svf.train()
#     elif method == "SVF":
#         svf = SVFC(method, inputs, outputs, data, c, eps, d)
#         svf.train()
#     else:
#         raise RuntimeError("The method selected doesn't exist")
#     return svf

# def modify_model(obj_SVF, c, eps):
#     """Método que devuelve un modelo SVF en docplex modificado
#
#     Args:
#         obj_SVF (svf_package.svf.SVF): Modelo a modificar
#         c (float): Valor del hiperparámetro C a modificar
#         eps (float): Valor del hiperparámetro épsilon a modificar
#     Returns:
#         svf_package.svf.SVF: Modelo SVF modificado
#     """
#     model = obj_SVF.modify_model(c, eps)
#     return model

def calculate_mse(svf, data_test):
        """Función que calcula el Mean Square Error (MSE) del cross-validation

        Args:
            data_test (pandas.DataFrame): conjunto de datos de test sobre los que se va a evaluar el MSE
            svf (svf_package.svf.SVF): modelo SVF sobre el que se va a evaluar los datos de test. Contiene los pesos (w) y el grid para calcular la estimación

        Returns:
            float: Mean Square Error obtenido para ese modelo y conjunto de datos
        """
        data_test_X = data_test.filter(svf.inputs)
        data_test_Y = data_test.filter(svf.outputs)
        n_out = len(data_test_Y.columns)
        error = 0
        n_obs_test = len(data_test_X)
        for i in range(n_obs_test):
            dmu = data_test_X.iloc[i]
            y_est = svf.get_estimation(dmu)
            print(y_est)
            for j in range(n_out):
                y = data_test_Y.iloc[i, j]
                error_obs = (y - y_est[j]) ** 2
                error = error + error_obs
        mse = error / n_obs_test
        return mse

def set_SVF_method(method, inputs, outputs, data, c, eps, d):
    if method == "SVF-SP":
        svf = SVF_SP(method, inputs, outputs, data, c, eps, d)
    elif method == "SSVF":
        svf = SSVF(method, inputs, outputs, data, c, eps, d)
    elif method == "SVF":
        svf = SVFC(method, inputs, outputs, data, c, eps, d)
    else:
        raise RuntimeError("The method selected doesn't exist")
    return svf
