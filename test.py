from NN import *
from Loss import *
from Activation import *
from Dataset import *
from Constraints import *

import os, sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def newNet():
    opti = csd.Opti()
    l1= Layer(1,3,opti,Logistic())
    l2 = Layer(3,1,opti,Logistic())
    net = NN([l1,l2],opti)
    return net

x_ = np.arange(-2,2,0.1)
y_ =getAns(x_,1.5,1)
N_repeat =10

f_inv_l = [f_inv_1,f_inv_x,f_inv_x2,f_inv_neg_relu,f_inv_logistic,f_inv_01]

res_l =[]
# for N_sample in [32,48,64,92,140,180,260,320]:
for N_sample in [32,48]:
    info1=0
    y_pred1=0
    info2=0;
    y_pred2=0
    for i in range(N_repeat):
        print(i)
        np.random.seed(i)
        x_train,y_train = dataset(N_sample,1.5,1)
        x_test,y_test = dataset(N_sample,1.5,1)
#         print("N_sample:",N_sample)
#         print("#### origin ####")
        net = newNet();
        loss = calLoss(x_train,y_train,net,l2_loss)
        with HiddenPrints():
            net.optimize(loss,[],10000)
        info,y_pred=test(x_test,y_test,net,1.5,1,False)
        info1+=info
        y_pred1+=y_pred
#         print("#### with predicate ####")
        net = newNet();
    #     loss = calLoss(x_train,y_train,net,l2_loss)
        
        cons = [calCons(x_train,y_train,net,f_inv__) for f_inv__ in f_inv_l]
    #     loss = calLoss(x_train,y_train,net,l2_loss)
        loss = calLoss(x_train,y_train,net,l2_loss)
        with HiddenPrints():
            net.optimize(loss,cons,10000)
        info,y_pred=test(x_test,y_test,net,1.5,1,False)
        info2+=info
        y_pred2+=y_pred
#     info1 = info1/N_repeat
#     y_pred1 = y_pred1/N_repeat
#     info2 = info2/N_repeat
#     y_pred2 = y_pred2/N_repeat
#     print("N_sample:",N_sample)
#     print("#### origin ####")
#     printInfo(info1)
#     drawInfo(x_train,y_train,x_,y_,y_pred1)
#     print("#### with predicate ####")
#     printInfo(info2)
#     drawInfo(x_train,y_train,x_,y_,y_pred2)
    res_l.append({'sc':[info1,y_pred1],'wc':[info2,y_pred2],'data':[x_train,y_train],"N_sample":N_sample})
for res in res_l:
    N_sample = res['N_sample']
    info1,y_pred1=res["sc"]
    info2,y_pred2 = res["wc"]
    x_train,y_train=res["data"]
    info1 = info1/N_repeat
    y_pred1 = y_pred1/N_repeat
    info2 = info2/N_repeat
    y_pred2 = y_pred2/N_repeat
    print("N_sample:",N_sample)
    print("#### origin ####")
    printInfo(info1)
    drawInfo(x_train,y_train,x_,y_,y_pred1)
    print("#### with predicate ####")
    printInfo(info2)
    drawInfo(x_train,y_train,x_,y_,y_pred2)