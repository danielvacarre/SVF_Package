from svf_package.grid.svf_splines_grid import SVFSplinesGrid
from tests import data

if __name__ == '__main__':

    inputs = ["x1","x2"]
    outputs = ["y1"]
    d = 2

    grid = SVFSplinesGrid(data,inputs,outputs,d)
    grid.create_grid()

    print(grid.data_grid)