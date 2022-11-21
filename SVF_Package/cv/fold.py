class FOLD:
    """Clase fold para validación cruzada
    """

    def __init__(self, data_train, data_test, num):
        """
            Constructor de la clase fold

        Args:
            data_train (pandas.DataFrame): Conjunto de datos de entrenamiento del fold

            data_test (pandas.DataFrame): Conjunto de datos de test del fold

            num (int): número del fold
            
        """
        self.data_train = data_train
        self.data_test = data_test
        self.num = num
