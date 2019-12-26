from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
class A1:
    def train(self,x_train, y_train):
        param_grid={'C':[0.1,0.5,1,5,10,50,100],'gamma':np.logspace(-10,0,11,endpoint=True, base=2)}#np.logspace(-5, 5, 11, endpoint=True, base=10)}#,'gamma':np.logspace(-15,3,19,endpoint=True, base=2)}
        clf = GridSearchCV(SVC(kernel = 'rbf'),param_grid,cv = 4)
        clf = clf.fit(x_train, y_train)
        self.clf = clf
        print(clf.best_score_)
        print(clf.best_estimator_)
        y_train_pred= clf.predict(x_train)
        train_acc = float(accuracy_score(y_train,y_train_pred))
        return train_acc
        '''
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size= 0.75, random_state=0)
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
