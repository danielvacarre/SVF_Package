from svf_package.grid.svfgrid import SVFGrid
from tests import data

if __name__ == '__main__':

    inputs = ["x1","x2","y1"]
    outputs = ["y1","y2"]
    d = 2

    grid = SVFGrid(data, inputs, outputs, d)
    grid.create_grid()

    print(grid.data_grid)