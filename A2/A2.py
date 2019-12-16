#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
class A2:
    #def __init__(self,para=1):
    #    self.para = para

    def train(self,x_train, y_train):
    # Build Logistic Regression Model
    #c_min = 0.1
    #c_max = 10
    #c_gap = 0.1
    #loop = int((c_max - c_min)/c_gap)
    #try_para=np.arange(c_min,c_max,c_gap)
        param_grid={'C':[0.1,0.5,1,5,10,50,100],'kernel':['linear','rbf']}
        #loop = len(try_para)######################
        #list_acc = list(range(0,loop))
        #for i in range(0,loop):
        clf = GridSearchCV(SVC( gamma = 'scale'),param_grid,cv = 3)
        clf = clf.fit(x_train, y_train)
        #self.kernel = clf.best_params_['kernel']
        #self.C = clf.best_params_['C']
        #print(self.kernel)
        #print(clf.cv_results_)
        self.clf = clf
        y_train_pred= clf.predict(x_train)
        train_acc = float(accuracy_score(y_train,y_train_pred))
            #model = SVC(C=try_para[i], gamma='scale')
    # Train the model using the training sets
            #model.fit(x_train, y_train)
            #y_pred_val = model.predict(x_val)
            #val_acc = float(accuracy_score(y_val,y_pred_val))
            #list_acc[i] = val_acc
    #print('Accuracy on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
        #m=list_acc.index(max(list_acc))
#    return (c_min + m*c_gap)
        #self.para = try_para[m]####################
        #model = SVC(C=self.para,gamma='scale')
        #model.fit(x_train, y_train)
        #y_pred_train = model.predict(x_train)
        #train_acc = float(accuracy_score(y_train,y_pred_train))
        #self.x_train = x_train
        #self.y_train = y_train
        #return max(list_acc), train_acc
        return train_acc

    def test(self,x_test,y_test):
        y_test_pred = self.clf.predict(x_test)
        test_acc = float(accuracy_score(y_test,y_test_pred))
        return test_acc
        
