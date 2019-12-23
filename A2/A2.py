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

        param_grid={'C':[0.1,0.5,1,5,10,50,100],'gamma':np.logspace(-9,1,11,endpoint=True, base=10)}
        #loop = len(try_para)######################
        #list_acc = list(range(0,loop))
        #for i in range(0,loop):
        clf = GridSearchCV(SVC( kernel = 'rbf'),param_grid,cv = 2)
        clf = clf.fit(x_train, y_train)
        self.clf = clf
        print(clf.best_score_)
        y_train_pred= clf.predict(x_train)
        train_acc = float(accuracy_score(y_train,y_train_pred))
        return train_acc
        '''
        clf = SVC(C=1, kernel = 'rbf', gamma = 'scale')
        clf = clf.fit(x_train, y_train)
        self.clf = clf
        y_train_pred= clf.predict(x_train)
        train_acc = float(accuracy_score(y_train,y_train_pred))
        return train_acc
        '''
    def test(self,x_test,y_test):
        y_test_pred = self.clf.predict(x_test)
        test_acc = float(accuracy_score(y_test,y_test_pred))
        return test_acc
