from itertools import product, zip_longest

import numpy
from numpy import arange, transpose
from pandas import DataFrame


class SVF_GRID:

    def __init__(self, data, inputs, d):
        self.data = data
        self.inputs = inputs
        self.d = d
        self.df_grid = None
        self.knot_list = None

    def create_grid(self):
        """
            Función que crea un grid en base a unos datos e hiperparámetro D
        """

        self.df_grid = DataFrame(columns=["id_cell","value","phi"])
        x = self.data.filter(self.inputs)
        # Numero de columnas x
        n_dim = len(x.columns)
        # Lista de listas de knot
        knot_list = list()
        # Lista de indices (posiciones) para crear el vector de subind
        knot_index = list()
        for col in range(0, n_dim):
            # knots de la dimension col
            knot = list()
            knot_max = x.iloc[:, col].max()
            knot_min = x.iloc[:, col].min()
            amplitud = (knot_max - knot_min) / self.d
            for i in range(0, self.d + 1):
                knot_i = knot_min + i * amplitud
                knot.append(knot_i)

            knot_list.append(knot)
            knot_index.append(arange(0, len(knot)))
        self.df_grid["id_cell"] = list(product(*knot_index))
        self.df_grid["value"] = list(product(*knot_list))
        self.knot_list = knot_list
        self.calculate_df_grid_phi()

    def search_observation(self, obs):
        """
            Función que devuelve la celda en la que se encuentra una observación en el grid
        Args:
            obs (list): Observación a buscar en el grid
        Returns:
            position (list): Vector con la posición de la observación en el grid
        """
        position = list()
        r = transpose(self.knot_list)
        for l in range(0, len(self.knot_list)):
            for m in range(0, len(self.knot_list[l])):
                trans = self.transformation(obs[l], r[m][l])
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

    def calculate_phi_observation(self,position):
        """
            Función que calcula el valor de la transformación (phi) de una observación en el grid.
        Args:
            position (list): Posición de la observación en el grid

        Returns:
            phi: Vector de 1 0 con la transformación del vector en base al grid 
        """
        phi = []
        n_dim = len(position)
        for i in range(0, len(self.df_grid)):
            for j in range(0, n_dim):
                if position[j] >= self.df_grid["id_cell"][i][j]:
                    value = 1
                else:
                    value = 0
                    break
            phi.append(value)
        return phi

    def transformation(self, x_i, t_k):
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

    def calculate_df_grid_phi(self):
        x = self.df_grid["value"]
        x_list = x.values.tolist()
        phi_list = list()
        for x in x_list:
            p = self.search_observation(x)
            phi = self.calculate_phi_observation(p)
            phi_list.append(phi)
        self.df_grid["phi"] = phi_list
