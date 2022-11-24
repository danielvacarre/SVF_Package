from svf_package.ssvf import SSVF
from svf_package.svf_splines import SVF_SP
from svf_package.svfc import SVFC


def train(method, inputs, outputs, data, c, eps, d):
    """ Método para entrenar el modelo generado. Se selecciona el algoritmo seleccionado y se entrena.

    Args:
        method (string): Método SVF_Methods que se quiere utilizar
        inputs (list): Inputs a evaluar en el conjunto de dato
        outputs (list): Outputs a evaluar en el conjunto de datos
        data (pandas.DataFrame): Conjunto de datos a evaluar
        c (float): Valor del hiperparámetro C que queremos evaluar
        eps (float): Valor del hiperparámetro épsilon que queremos evaluar
        d (int): Valor del hiperparámetro d que queremos evaluar

    Raises:
        RuntimeError: Indica que el método no existe

    Returns:
        svf_package.svf.SVF: Devuelve un modelo SVF del método escogido
    """

    if method == "SVF-SP":
        svf = SVF_SP(method, inputs, outputs, data, c, eps, d)
        svf.train()
    elif method == "SSVF":
        svf = SSVF(method, inputs, outputs, data, c, eps, d)
        svf.train()
    elif method == "SVF":
        svf = SVFC(method, inputs, outputs, data, c, eps, d)
        svf.train()
    else:
        raise RuntimeError("The method selected doesn't exist")
    return svf

def modify_model(obj_SVF, c, eps):
    """Método que devuelve un modelo SVF en docplex modificado

    Args:
        obj_SVF (svf_package.svf.SVF): Modelo a modificar
        c (float): Valor del hiperparámetro C a modificar
        eps (float): Valor del hiperparámetro épsilon a modificar
    Returns:
        svf_package.svf.SVF: Modelo SVF modificado
    """
    model = obj_SVF.modify_model(c, eps)
    return model
