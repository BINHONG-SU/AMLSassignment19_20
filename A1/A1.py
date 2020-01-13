from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

class A1:
    def train(self,x_train, y_train):
        param_grid={'C':[1,5,10,50,100,200],'gamma':np.logspace(-20,-15,6,endpoint=True, base=2)}   #Set the hyperparameters need to be trained
        clf = GridSearchCV(SVC(kernel = 'rbf'),param_grid,cv = 4)   #Use SVM with rbf kernel and 4-Fold Cross validation, consider all hyperparameter combinations
        clf.fit(x_train, y_train)   #Train the models and find the best estimator
        print('A1 best estimator',clf.best_estimator_)
        print('acc_A1_val',clf.best_score_)
        self.model = clf.best_estimator_
        y_train_pred= self.model.predict(x_train)   #the prediction of train set based on the selected model
        train_acc = float(accuracy_score(y_train,y_train_pred))
        return train_acc

    def test(self,x_test,y_test):
        y_test_pred = self.model.predict(x_test)    #Predict the label of testing set based on the selected model
        test_acc = float(accuracy_score(y_test,y_test_pred))
        return test_acc


    def plot_learning_curve(self,x_train,y_train):
        train_sizes, train_scores, test_scores=learning_curve(self.model, x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=2)  #get the training scores, testing scores of estimator, use 5-fold cross validation
        train_scores_mean = np.mean(train_scores, axis=1)   #get the mean value of training scores
        test_scores_mean = np.mean(test_scores, axis=1)     #get the mean value of testing scores
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r',label='SVC_train') #plot the training scores
        plt.plot(train_sizes, test_scores_mean, 'o-', color='b',label='SVC_test')   #plot the testing scores
        plt.legend(('Train accuracy', 'Test accuracy'), loc='lower right')
        plt.title('Learning curve of A1_best_estimator_')
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.grid()
        plt.show()
        return None
