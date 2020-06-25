import numpy as nu
import casadi as csd

def l2_loss(y1,y2):
    return (y1-y2)**2

def calLoss(x_train,y_train,net,loss_func,numerical=False):
    loss = 0

    for x_i,y_i in zip(x_train,y_train):
    #     loss+=(f_forward((w,b),x_i)-y_i)**2
    #     loss+=l2_loss(f_forward((w,b),x_i),y_i)
        loss+=loss_func(net.forward(x_i,numerical),y_i)
    return loss
def calAcc(x_test,y_test,net):
    cnt = 0
    for x_i,y_i in zip(x_test,y_test):
        if(net.forward(x_i,True)>0.5):
            val = 1
        else:
            val=0
        if(val==y_i):
            cnt+=1
    return cnt/len(x_test)