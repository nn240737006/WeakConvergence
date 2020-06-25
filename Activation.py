import casadi as csd
import numpy as np
class Activation(object):
    def __init__(self,):
        pass
    def cal(self,x):
        pass
class Logistic(Activation):
    def __init__(self,):
        x = csd.SX.sym('x')
        # y = csd.SX.sym('y')
        exp_x =csd.exp(x)
#         self.f = csd.Function("f",[x],[exp_x/(1+exp_x)])
        self.f = csd.Function("f",[x],[exp_x/(1+exp_x)])
    def cal(self,x,style="casadi"):
#         return self.f(x)
        if(style=="casadi"):
            exp_x =csd.exp(x)
        elif(style=="numpy"):
            exp_x = np.exp(x)
        return exp_x/(1+exp_x) 
class Relu(Activation):
    def __init__(self,):
        x = csd.SX.sym('x')
        # y = csd.SX.sym('y')
        exp_x =csd.exp(x)
#         self.f = csd.Function("f",[x],[exp_x/(1+exp_x)])
        self.f = csd.Function("f",[x],[exp_x/(1+exp_x)])
    def cal(self,x,style="casadi"):
#         return self.f(x)
        if(style=="casadi"):
            return csd.fmax(0,x)
        elif(style=="numpy"):
            return max(0,x)