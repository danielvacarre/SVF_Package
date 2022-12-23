from svf_package.cv.cv import CrossValidation
from tests import data

if __name__ == '__main__':

    inputs = ["x1"]
    outputs = ["y1"]
    C = [1,10]
    eps = [0,10]
    D = [2,3]
    # method = "SSVF"
    # method = "SVF"
    method = "SVF-SP"
    method = "dual"

    cv_obj = CrossValidation(method, inputs, outputs, data, C, eps, D, verbose=True)
    cv_obj.cv()

    # svf = train(method, inputs, outputs, data, c, eps, d)
    # #print(svf.model.export_to_string())
    # svf.solve()
    # #print(svf.solution.w)
    # print(svf.estimation([3]))
    #
    # svf.model = svf.modify_model(100, 10)
    # #print(svf.model.export_to_string())
    #
    # inputs = ["x1"]
    # outputs = ["y1","y2"]
    # c = 1
    # eps = 0
    # d = 2
    #
    # svf = train(method, inputs, outputs, data, c, eps, d)
    #
    # #print(svf.model.export_to_string())
    # svf.solve()
    # #print(svf.solution.w)
    # print(svf.estimation([3]))
    #
    # svf.model = svf.modify_model(100, 10)
    # #print(svf.model.export_to_string())
    #
    # inputs = ["x1","x2"]
    # outputs = ["y1"]
    # c = 1
    # eps = 0
    # d = 2
    #
    # svf = train(method, inputs, outputs, data, c, eps, d)
    #
    # #print(svf.model.export_to_string())
    # svf.solve()
    # #print(svf.solution.w)
    # print(svf.estimation([1,3]))
    #
    # svf.model = svf.modify_model(100, 10)
    # #print(svf.model.export_to_string())
    #
    # inputs = ["x1","x2"]
    # outputs = ["y1","y2"]
    # c = 1
    # eps = 0
    # d = 2
    #
    # svf = train(method, inputs, outputs, data, c, eps, d)
    #
    # #print(svf.model.export_to_string())
    # svf.solve()
    # #print(svf.solution.w)
    # print(svf.estimation([1,3]))
    #
    # svf.model = svf.modify_model(100, 10)
    # #print(svf.model.export_to_string())
    #
    # inputs = ["x1","x2","y1"]
    # outputs = ["y1"]
    # c = 1
    # eps = 0
    # d = 2
    #
    # svf = train(method, inputs, outputs, data, c, eps, d)
    #
    # #print(svf.model.export_to_string())
    # svf.solve()
    # #print(svf.solution.w)
    # print(svf.estimation([1,2,3]))
    #
    # svf.model = svf.modify_model(100, 10)
    # #print(svf.model.export_to_string())
    #
    # inputs = ["x1","x2","y1"]
    # outputs = ["y1","y2"]
    # c = 1
    # eps = 0
    # d = 2
    #
    # svf = train(method, inputs, outputs, data, c, eps, d)
    #
    # # print(svf.model.export_to_string())
    # svf.solve()
    # #print(svf.solution.w)
    # print(svf.estimation([1,2,3]))
    #
    # svf.model = svf.modify_model(100, 10)
    # #print(svf.model.export_to_string())