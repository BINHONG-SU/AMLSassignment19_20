import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

class B2:
    def train(self,x_train, y_train,x_val,y_val):
        try_C=np.logspace(-10, -4, 7, endpoint=True, base=2)  #Set the hyperparameters need to be trainedS
        iter_C = len(try_C)
        val_acc_max = 0

        for i in range(0,iter_C):
            clf = LinearSVC(C = try_C[i], max_iter =99999)#, tol=0.001)   #Use a LinearSVC model
            clf.fit(x_train, y_train)    #Train the model
            y_val_pred= clf.predict(x_val)  #Predict the labels of validation input data based on the model
            val_acc = float(accuracy_score(y_val,y_val_pred))
            if val_acc > val_acc_max:   #save the best estimator
                val_acc_max = val_acc
                self.model = clf
        print('B2 best estimator',self.model)
        self.model.fit(x_train, y_train)
        y_train_pred = self.model.predict(x_train)  #Predict the labels of training input data based on the best estimator
        train_acc = float(accuracy_score(y_train,y_train_pred))
        return val_acc_max, train_acc

    def test(self,x_test,y_test):
        y_test_pred = self.model.predict(x_test)    #Predict the label of testing input data based on the best estimator
        test_acc = float(accuracy_score(y_test,y_test_pred))
        return test_acc

    def plot_learning_curve(self,x_train,y_train):
        train_sizes, train_scores, test_scores=learning_curve(self.model, x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=2)#get the training scores, testing scores of estimator, use 5-fold cross validation
        train_scores_mean = np.mean(train_scores, axis=1)    #get the mean value of training scores
        test_scores_mean = np.mean(test_scores, axis=1)     #get the mean value of testing scores
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r',label='SVC_train') #plot the training scores
        plt.plot(train_sizes, test_scores_mean, 'o-', color='b',label='SVC_test')    #plot the testing scores
        plt.legend(('Train accuracy', 'Test accuracy'), loc='lower right')
        plt.title('Learning curve of B2_best_estimator_')
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.grid()
        plt.show()
        return None
