from numpy import arange
from SVF_Package.grid.grid import GRID


class SVFSplinesGrid(GRID):

    def __init__(self, data, inputs, outputs, d):
        """
            Constructor de la clase SVFSplinesGrid
        Args:
            data (pandas.DataFrame): conjunto de datos sobre los que se construye el grid
            inputs (list): listado de inputs
            d (int): número de particiones en las que se divide el grid
        """
        super().__init__(data, inputs, outputs, d)

    def create_grid(self):
        """
            Función que crea un grid en base a unos datos e hiperparámetro d
        """
        x = self.data.filter(self.inputs)
        # Numero de columnas x
        n_dim = len(x.columns)
        # Lista de listas de knot
        knot_list = list()
        # Lista de indices (posiciones) para crear el vector de subind
        knot_index = list()
        for col in range(0, n_dim):
            # knots de la dimension col
            knot = [0]
            knot_max = x.iloc[:, col].max()
            knot_min = x.iloc[:, col].min()
            amplitud = (knot_max - knot_min) / self.d
            for i in range(0, self.d + 1):
                knot_i = knot_min + i * amplitud
                knot.append(knot_i)
            knot_list.append(knot)
            knot_index.append(arange(0, len(knot)))
        self.knot_list = knot_list
        self.calculate_data_grid()

    def calculate_dmu_phi(self, dmu):
        """
            Función que calcula el valor de la transformación (phi) de una observación en el grid.
        Args:
            dmu (list): Observación a evaluar

        Returns:
            list: Vector de 1 0 con la transformación del vector en base al grid
        """
        phi_list = list()
        dmu_phi = list()
        n_dim = len(dmu)
        for j in range(n_dim):
            phi = [1]
            for i in range(len(self.knot_list[j])):
                if dmu[j] > self.knot_list[j][i]:
                    value = dmu[j] - self.knot_list[j][i]
                else:
                    value = 0
                phi.append(value)
            phi_list.append(phi)
        for i in range(len(self.outputs)):
            dmu_phi.append(phi_list)
        return dmu_phi

    def calculate_data_grid(self):
        """Método para añadir al dataframe grid el valor de la transformada de cada observación
        """
        self.data_grid = self.data.copy()
        dmu_list = self.data_grid.filter(self.inputs)
        dmu_values_list = dmu_list.values.tolist()
        phi_list = list()
        for dmu_values in dmu_values_list:
            phi = self.calculate_dmu_phi(dmu_values)
            phi_list.append(phi)
        self.data_grid["phi"] = phi_list
