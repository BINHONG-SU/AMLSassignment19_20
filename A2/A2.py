from sklearn import svm
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
class A2:
    #def __init__(self,para=1):
    #    self.para = para

    def train(self,x_train, y_train):
    # Build Logistic Regression Model

        param_grid={'C':[0.1,0.5,1,5,10,50,100],'gamma':np.logspace(-10,0,11,endpoint=True, base=2)}
        #loop = len(try_para)######################
        #list_acc = list(range(0,loop))
        #for i in range(0,loop):
        clf = GridSearchCV(SVC( kernel = 'rbf'),param_grid,iid='True',cv = 4)
        clf = clf.fit(x_train, y_train)
        self.clf = clf
        print(clf.best_score_)
        print(clf.best_estimator_)
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

    def plot_learning_curve(self,x_train,y_train):
        train_sizes, train_scores, test_scores=learning_curve(self.clf, x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r',label='SVC_train')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='b',label='SVC_test')
        plt.legend(('Train accuracy', 'Test accuracy'), loc='lower right')
        plt.title('Learning curve of A2_best_estimator_')
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.grid()
        plt.show()
        return None
