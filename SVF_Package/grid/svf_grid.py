from itertools import product
from numpy import arange, asarray, float32
from pandas import DataFrame

from svf_package.grid.grid import GRID


class SVF_GRID(GRID):
    """
        Clase generadora de un grid SVF. Sirve tanto para SVF como SSVF
    """

    def __init__(self, data, inputs, d):
        """
            Constructor de la clase SVF_GRID
        Args:
            data (pandas.DataFrame): conjunto de datos sobre los que se construye el grid
            inputs (list): listado de inputs
            d (list): número de particiones en las que se divide el grid
        """
        super().__init__(data, inputs, d)

    def create_grid(self):
        """
            Función que crea un grid en base a unos datos e hiperparámetro d
        """

        self.df_grid = DataFrame(columns=["id_cell", "value", "phi"])
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
        self.calculate_df_grid()
        self.calculate_data_grid()

    def calculate_phi_observation(self, position):
        """
            Función que calcula el valor de la transformación (phi) de una observación en el grid.
        Args:
            position (list): Posición de la observación en el grid

        Returns:
            list: Vector de 1 0 con la transformación del vector en base al grid
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

    def calculate_df_grid(self):
        """Método para añadir al dataframe grid el valor de la transformada de cada observación
        """
        x = self.df_grid["value"]
        x_list = x.values.tolist()
        phi_list = list()
        c_cells = list()
        for x in x_list:
            p = self.search_observation(x)
            phi = self.calculate_phi_observation(p)
            phi_list.append(phi)
        for index,cell in self.df_grid.iterrows():
            c_cell = search_contiguous_cell(cell['id_cell'])
            c_cells.append(c_cell)
        self.df_grid["phi"] = phi_list
        self.df_grid["c_cells"] = c_cells

    def calculate_data_grid(self):
        """Método para añadir al dataframe grid el valor de la transformada de cada observación
        """
        self.data_grid = self.data.copy()
        x = self.data_grid.filter(self.inputs)
        x_list = x.values.tolist()
        phi_list = list()
        c_cells = list()
        for x in x_list:
            p = self.search_observation(x)
            phi = self.calculate_phi_observation(p)
            phi_list.append(phi)
            c_cell = search_contiguous_cell(p)
            c_cells.append(c_cell)
        self.data_grid["phi"] = phi_list
        self.data_grid["c_cells"] = c_cells

def search_contiguous_cell(cell):
    con_c_list = list()
    cell = list(cell)
    for dim in range(len(cell)):
        value = (cell[dim]) - 1
        con_cell = cell.copy()
        if value >= 0:
            con_cell[dim] = value
            con_c_list.append(tuple(con_cell))
    return con_c_list


