from numpy import transpose


class GRID:

    """
        Clase grid sobre el que se realiza el módelo SVF. Un grid es una partición del espacio de los inputs que está divido por celdas
    """

    def __init__(self, data, inputs, d):
        """Constructor de la clase grid

        Args:
            data (pandas.DataFrame): conjunto de datos sobre los que se construye el grid
            inputs (list): listado de inputs
            d (list): número de particiones en las que se divide el grid
        """
        self.data = data
        self.inputs = inputs
        self.d = d
        self.df_grid = None
        self.knot_list = None

    def search_observation(self, dmu):
        """
            Función que devuelve la celda en la que se encuentra una observación en el grid
        Args:
            dmu (list): Observación a buscar en el grid
        Returns:
            position (list): Vector con la posición de la observación en el grid
        """
        position = list()
        r = transpose(self.knot_list)
        for l in range(0, len(self.knot_list)):
            for m in range(0, len(self.knot_list[l])):
                trans = self.transformation(dmu[l], r[m][l])
                if trans < 0:
                    position.append(m - 1)
                    break
                if trans == 0:
                    position.append(m)
                    break
                if trans > 0 and m == len(self.knot_list[l]) - 1:
                    position.append(m)
                    break
        return position