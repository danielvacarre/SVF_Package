from numpy import transpose


class GRID:

    def __init__(self, data, inputs, d):
        self.data = data
        self.inputs = inputs
        self.d = d
        self.df_grid = None
        self.knot_list = None

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