import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
class B2:
    def train(self,x_train, y_train,x_val,y_val):
        try_C=np.logspace(-3, 10, 14, endpoint=True, base=2)#,try_C=np.logspace(-13, 1, 15, endpoint=True, base=2)
        iter_C = len(try_C)
        #try_gamma = np.logspace(-12,0,13,endpoint=True, base=2)
        #iter_gamma = len(try_gamma)
        val_acc_max = 0

        for i in range(0,iter_C):
            #for j in range(0,iter_gamma):
            #clf = LogisticRegression(solver='lbfgs',multi_class= 'multinomial',tol=0.001,C=try_C[i],max_iter=5000)
                clf = SVC( C = try_C[i], kernel = 'linear', decision_function_shape = 'ovo')
                clf.fit(x_train, y_train)
                print('C=',try_C[i])
                #print('gamma=',try_gamma[j])
                y_val_pred= clf.predict(x_val)
                val_acc = float(accuracy_score(y_val,y_val_pred))
                print('B2val_acc = ',val_acc)
                print(' ')
                if val_acc > val_acc_max:
                    val_acc_max = val_acc
                    self.model = clf
        self.model.fit(x_train, y_train)
        y_train_pred = self.model.predict(x_train)
        train_acc = float(accuracy_score(y_train,y_train_pred))
        return val_acc_max, train_acc

    def test(self,x_test,y_test):
        y_test_pred = self.model.predict(x_test)
        test_acc = float(accuracy_score(y_test,y_test_pred))
        return test_acc

    def plot_learning_curve(self,x_train,y_train):
        train_sizes, train_scores, test_scores=learning_curve(self.model, x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r',label='SVC_train')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='b',label='SVC_test')
        plt.legend(('Train accuracy', 'Test accuracy'), loc='lower right')
        plt.title('Learning curve of B2_best_estimator_')
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.grid()
        plt.show()
        return None
