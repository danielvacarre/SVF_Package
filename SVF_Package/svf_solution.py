class SVFSolution:
    """Clase solución de SVF
    """
    def __init__(self, w, xi):
        """Constructor de la clase solución SVF

        Args:
            w (list): Valores de los pesos w del modelo ya solucionado
            xi (list): Valores xi del modelo ya solucionado
        """
        self.w = w
        self.xi = xi
