class SVFDualSolution:
    """Clase solución de SVF
    """
    def __init__(self, gamma, alpha, delta, w):

        """Constructor de la clase solución SVF en los forma dual

        Args:
            gamma (list): Valores de la variable gamma del modelo ya solucionado
            alpha (list): Valores de la variable alpha del modelo ya solucionado
            delta (list): Valores de la variable delta del modelo ya solucionado
            w (list): Valores de los pesos w del modelo ya solucionado
        """

        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.w = w
