# import casadi as csd
import numpy as np
import matplotlib.pyplot as plt
from Loss import calLoss,l2_loss,calAcc
def dataset(N_sample,m,v):
    N_sample=int(N_sample/2)
#     x_range = (-4,4)
#     x_train = np.arange(x_range[0],x_range[1],(x_range[1]-x_range[0])/N_sample)
#     # y_train = -x_train**2
#     y_train = Logistic().cal(x_train,"numpy")+np.random.rand(*x_train.shape)*0.01
    gus=np.random.normal
    x_train = np.concatenate((gus(-m,v,N_sample),gus(m,v,N_sample)))
    y_train = np.concatenate((np.zeros(N_sample)+1,np.zeros(N_sample)+0))
#     plt.scatter(x_train,y_train)
    return x_train,y_train
def printInfo(info):
    loss_new,fit_loss,acc_err = info
    print("loss_new:",loss_new)
    print("fit_loss:",fit_loss)
    print("acc_err:",acc_err)

def f_gus(x,m,r):
    return np.exp(-(((x-m)/(r*1))**2)/2)
# x_ = np.arange(-1,1,0.01)
def getAns(x,m,v):
    y1=f_gus(x,-m,v)
    y2=f_gus(x,m,v)
    temp=y1-y2
    temp[temp<0]=0
    y_=y1/(y1+y2)
    return y_
# def drawInfo(x_train,y_train,x_,y_,y_pred):
def drawInfo(x_train,y_train,x_,y_,y_pred):
    fig = plt.figure()
    plt.scatter(x_train,y_train)
    plt.plot(x_,y_)
    plt.plot(x_,y_pred)
    plt.show()
def test(x_train,y_train,net,m,v,draw=True,x_=None,y_=None):
    if(x_==None or y_==None):
        x_ = np.arange(-2,2,0.1)
        y_ =getAns(x_,m,v)
    loss_new = calLoss(x_train,y_train,net,l2_loss,True)
#     loss_new = calLoss(x_,y_,net,l2_loss,True)
    acc = calAcc(x_train,y_train,net)
    fit_loss = calLoss(x_,y_,net,l2_loss,True)
    y_pred = np.array([ float(net.forward(np.array([[x__]]),True))  for x__ in x_])
    if(draw):
        print("N_sample:",len(x_train))
        print("loss_new:",loss_new)
        print("fit_loss:",fit_loss)
        print("acc_err:",1-acc)
        fig = plt.figure()
        plt.scatter(x_train,y_train)
        plt.plot(x_,y_)
        plt.plot(x_,y_pred)
#     display(plt)
        plt.show()
    return np.array([loss_new,fit_loss,1-acc]),y_pred
