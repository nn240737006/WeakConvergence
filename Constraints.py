from Activation import Logistic
import casadi as csd
def calCons(x_train,y_train,net,f_inv):
    left = 0
    right = 0
    N_ = len(x_train)
    for x_i ,y_i in zip(x_train,y_train):
        left += net.forward(x_i)*f_inv(x_i)/N_
        right += y_i*f_inv(x_i)/N_
    return left==right
def f_inv_1(x):
    return 1
def f_inv_x(x):
    return x
def f_inv_x2(x):
    return x**2
def f_inv_neg_relu(x):
    return csd.fmax(0,-x)
def f_inv_relu(x):
    return csd.fmax(0,x)
def f_inv_logistic(x):
    return 1-Logistic().cal(x)
def f_inv_01(x):
    return (x-1)*(x-0)*4