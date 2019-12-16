import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
class B2:
    def train(self,x_train, y_train,x_val,y_val):
        try_C=np.logspace(-5, 15, 21, endpoint=True, base=2)
        iter_C = len(try_C)
        try_gamma = np.logspace(-15,3,19,endpoint=True, base=2)
        iter_gamma = len(try_gamma)
        val_acc_max = 0
        for i in range(0,iter_C):
            for j in range(0,iter_gamma):
            #clf = LogisticRegression(solver='lbfgs',multi_class= 'multinomial',tol=0.001,C=try_para[i],max_iter=5000)
                clf = SVC( C = try_C[i], kernel = 'rbf', gamma = try_gamma[j], decision_function_shape = 'ovo')
                clf.fit(x_train, y_train)
                print('C=',try_C[i])
                print('gamma=',try_gamma[j])
                y_val_pred= clf.predict(x_val)
                val_acc = float(accuracy_score(y_val,y_val_pred))
                print('val_acc = ',val_acc)
                print(' ')
                if val_acc > val_acc_max:
                    val_acc_max = val_acc
                    self.model = clf
            #list_acc[i] = val_acc
        #m=list_acc.index(max(list_acc))
        #print(try_para[m])
        #model = SVC(C=try_para[m], kernel = 'rbf', gamma = 'scale', decision_function_shape = 'ovo')
        #model = LogisticRegression(solver='lbfgs',multi_class= 'multinomial', tol=0.001,C=try_para[m],max_iter=5000)
        self.model.fit(x_train, y_train)
        #self.model = model
        y_train_pred = self.model.predict(x_train)
        train_acc = float(accuracy_score(y_train,y_train_pred))
            #self.x_train = x_train
            #self.y_train = y_train
        return val_acc_max, train_acc
        #return max(list_acc), train_acc
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

    def test(self,x_test,y_test):
        y_test_pred = self.model.predict(x_test)
        test_acc = float(accuracy_score(y_test,y_test_pred))
        return test_acc
