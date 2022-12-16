from pandas import concat
from svf_package.methods.ssvf import SSVF
from svf_package.methods.svf_dual import SVFDual
from svf_package.methods.svf_splines import SVFSplines
from svf_package.methods.svfc import SVFC

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
        for j in range(n_out):
            y = data_test_Y.iloc[i, j]
            error_obs = (y - y_est[j]) ** 2
            error = error + error_obs
    mse = error / n_obs_test
    return mse

def create_SVF(method, inputs, outputs, data, c, eps, d):
    """Función que crea un objeto del tipo SVF en función del método que se selecciona
    Args:
        method (string): Método SVF que se quiere utilizar
        inputs (list): Inputs a evaluar en el conjunto de dato
        outputs (list): Outputs a evaluar en el conjunto de datos
        data (pandas.DataFrame): Conjunto de datos a evaluar
        c (float): Valores del hiperparámetro C del modelo
        eps (float): Valores del hiperparámetro épsilon del modelo
        d (int): Valor del hiperparámetro d del modelo

    Raises:
        RuntimeError: Indica que no existe el método seleccionado

    Returns:
        object: Devuelve un objeto del método SVF seleccionado
    """
    data = create_dataset(inputs, outputs, data)
    if method == "SVF-SP":
        svf = SVFSplines(method, inputs, outputs, data, c, eps, d)
    elif method == "SSVF":
        svf = SSVF(method, inputs, outputs, data, c, eps, d)
    elif method == "SVF":
        svf = SVFC(method, inputs, outputs, data, c, eps, d)
    elif method == "dual":
        svf = SVFDual(method, inputs, outputs, data, c, eps, d)
    else:
        raise RuntimeError("The method selected doesn't exist")
    return svf

def create_dataset(inputs,outputs, data):
    for input in inputs:
        if input not in data.columns:
            raise RuntimeError(input + " is not in the dataset")
    for output in outputs:
        if output not in data.columns:
            raise RuntimeError(output + " is not in the dataset")
    df_inputs = data.filter(inputs)
    df_outputs = data.filter(outputs)
    dataset = concat([df_inputs, df_outputs],axis=1)
    return dataset