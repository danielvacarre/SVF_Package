from numpy import transpose


class GRID:
    """
        Clase grid sobre el que se realiza el módelo SVF. Un grid es una partición del espacio de los inputs que está divido por celdas
    """

    def __init__(self, data, inputs, outputs, d):
        """Constructor de la clase grid

        Args:
            data (pandas.DataFrame): conjunto de datos sobre los que se construye el grid
            inputs (list): listado de inputs
            outputs (list): listado de outputs
            d (list): número de particiones en las que se divide el grid
        """
        self.data = data
        self.inputs = inputs
        self.outputs = outputs
        self.d = d
        self.data_grid = None
        self.knot_list = None

    def search_dmu(self, dmu):
        """
            Función que devuelve la celda en la que se encuentra una observación en el grid
        Args:
            dmu (list): Observación a buscar en el grid
        Returns:
            position (list): Vector con la posición de la observación en el grid
        """
        cell = list()
        r = transpose(self.knot_list)
        for l in range(0, len(self.knot_list)):
            for m in range(0, len(self.knot_list[l])):
                trans = transformation(dmu[l], r[m][l])
                if trans < 0:
                    cell.append(m - 1)
                    break
                if trans == 0:
                    cell.append(m)
                    break
                if trans > 0 and m == len(self.knot_list[l]) - 1:
                    cell.append(m)
                    break
        return tuple(cell)


def transformation(x_i, t_k):
    """
    Funcion que evalua si el valor de una observación es mayor o menor al de un nodo del grid.
    Si es mayor devuelve 1, si es igual devuelve 0 y si es menor devuelve -1.

    Args:
        x_i (float) : Valor de la celda a evaluar

        t_k (float) : Valor del nodo con el que se quiere comparar

    Returns:
        res (int): Resultado de la transformacion
    """

    z = x_i - t_k
    if z < 0:
        return -1
    elif z == 0:
        return 0
    else:
        return 1
