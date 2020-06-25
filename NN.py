import casadi as csd
import numpy as np

class Layer(object):
    def __init__(self,input_size,output_size,opti,activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.opti =opti
        self.w_size = (output_size,input_size)
        self.b_size = (output_size,1)
        self.w = self.opti.variable(*self.w_size)
        self.b = self.opti.variable(*self.b_size)
        self.w_n = 0
        self.b_n = 0
        self.w_init = np.random.rand(*self.w_size)*2-1
        self.b_init = np.random.rand(*self.b_size)*2-1
    def forward(self,x,numerical=False):
        if(numerical):
            if(type(x)!=np.ndarray and type(x)!=csd.casadi.DM):
#                 print(x,type(x))
                x = np.array([x]).reshape(self.input_size,1)
        if(not numerical):
            w = self.w
            b = self.b
        else:
            w = self.w_n
            b = self.b_n
        y_=w@x+b
        if(self.activation == None):
            return y_
        else:
            return self.activation.cal(y_)
class NN(object):
    def __init__(self,layers,opti):
        self.layers=layers
        self.opti = opti
    def Sequential(self,layers):
        self.layers=layers
    def forward(self,x,numerical=False):

        temp = x
        for i in range(len(self.layers)):
            temp = self.layers[i].forward(temp,numerical)
        return temp
    def __str__(self):
        s="NN:\n"
        for i,layer in enumerate(self.layers):
            s+=str(i)+":\n"
#             s+="w:"+str(self.sol.value(layer.w))+"\n"
#             s+="b:"+str(self.sol.value(layer.b))+"\n"
            s+="w:"+str(layer.w_n)+"\n"
            s+="b:"+str(layer.b_n)+"\n"
        return s
    def updateNumericalValue(self,sol):
        def setVal(v,layer,type_):
            t_size = type_+"_size"
            t_n = type_+"_n"
            if(type(v)!=np.ndarray):
                setattr(layer,t_n,np.array([v]).reshape(*getattr(layer,t_size)))
            else:
                setattr(layer,t_n,v.reshape(*getattr(layer,t_size)))
        for layer in self.layers:
            # o_sz =layer.output_size
            # i_sz = layer.input_size
#             layer.w_n = sol.value(layer.w).reshape(o_sz,i_sz)
#             layer.b_n = sol.value(layer.b).reshape(o_sz,1)
#             layer.w_n = sol.value(layer.w).reshape(*layer.w_size)
#             vb=sol.value(layer.b)
            setVal(sol.value(layer.w),layer,"w")
            setVal(sol.value(layer.b),layer,"b")
        #             if(type(vb)!=np.ndarray):
#                 layer.b_n = np.array([vb]).reshape(*layer.b_size)
#             else:
#                 layer.b_n = vb.reshape(*layer.b_size)
    def optiInit(self):
        for layer in self.layers:
            self.opti.set_initial(layer.w,layer.w_init)
            self.opti.set_initial(layer.b,layer.b_init)
    def setConstraints(self,constraints=[]):
        for cons in constraints:
            self.opti.subject_to(cons)
    def optimize(self,loss,constraints=[],maxIter=200):
        self.opti.minimize(loss)
        self.setConstraints(constraints)
        self.optiInit()
        # opti.subject_to( f_cons_eq_1(x,y) )
        # opti.subject_to(       f_cons_ieq_1(x,y) )
#         self.opti.solver('ipopt')
        p_opts = {"expand":True}
        s_opts = {"max_iter": maxIter}
        self.opti.solver("ipopt",p_opts,
                            s_opts)
        try:
            self.sol = self.opti.solve()
        except:
            self.sol = self.opti.debug
        self.updateNumericalValue(self.sol)