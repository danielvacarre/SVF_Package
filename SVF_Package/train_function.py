from svf_package.ssvf import SSVF
from svf_package.svf_splines import SVF_SP


def train(method, inputs, outputs, data, C, eps, d):
    """ Método para entrenar el modelo generado. Se selecciona el algoritmo seleccionado y se entrena.

    Args:
        method (string): Método SVF_Methods que se quiere utilizar
        inputs (list): Inputs a evaluar en el conjunto de dato
        outputs (list): Outputs a evaluar en el conjunto de datos
        data (pandas.DataFrame): Conjunto de datos a evaluar
        C (list): Valores del hiperparámetro C que queremos evaluar
        eps (list): Valores del hiperparámetro épsilon que queremos evaluar
        d (list): Valores del hiperparámetro d que queremos evaluar

    Raises:
        RuntimeError: Indica que el método no existe

    Returns:
        svf_package.svf.SVF: Devuelve un modelo SVF del método escogido
    """

    if method == "SVF-SP":
        print("SVF-SP")
        svf = SVF_SP(method, inputs, outputs, data, C, eps, d)
        svf.train()
    elif method == "SSVF":
        svf = SSVF(method, inputs, outputs, data, C, eps, d)
        svf.train()
    elif method == "SVF":
        pass
        # print("SVF")
        # self.train_svf()
    else:
        raise RuntimeError("The method selected doesn't exist")
    return svf

def modify_model(obj_SVF, c, eps):
    if obj_SVF.method == "SVF-SP":
        pass
        # print("SVF-SP")
        # self.train_svf_sp()
    elif obj_SVF.method == "SSVF":
        model = obj_SVF.modify_model(c, eps)
    elif obj_SVF.method == "SVF":
        pass
        # print("SVF")
        # self.train_svf()
    else:
        raise RuntimeError("The model cannot be modified")
    return model
