from svf_package.ssvf import SSVF

def train(method, inputs, outputs, data, C, eps, d):
    """
        Método para entrenar el modelo generado. Se selecciona el algoritmo seleccionado y se entrena.
    """
    if method == "SVF-SP" :
        pass
        # print("SVF-SP")
        # self.train_svf_sp()
    elif method == "SSVF":
        svf = SSVF(method, inputs, outputs, data, C, eps, d)
        svf.model = svf.train()
    elif method == "SVF":
        pass
        # print("SVF")
        #self.train_svf()
    else:
        raise RuntimeError("The method selected doesn't exist")
    return svf
