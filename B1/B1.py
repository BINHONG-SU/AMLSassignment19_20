from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class B1:

    def train(self,x_train, y_train,x_val,y_val):
        try_C=[0.1,0.5]#,1,5,10,50,100]
        loop = len(try_C)
        val_acc_max = 0
        list_acc = list(range(0,loop))
        for i in range(0,loop):
            clf = SVC( C= try_C[i], kernel = 'rbf', gamma = 'scale',decision_function_shape = 'ovo')
            clf.fit(x_train, y_train)
            y_val_pred= clf.predict(x_val)
            val_acc = float(accuracy_score(y_val,y_val_pred))
            if val_acc > val_acc_max:
                val_acc_max = val_acc
                self.model = clf
        self.model.fit(x_train, y_train)
        y_train_pred = self.model.predict(x_train)
        train_acc = float(accuracy_score(y_train,y_train_pred))
        return max(list_acc), train_acc


    def test(self,x_test,y_test):
        y_test_pred = self.model.predict(x_test)
        test_acc = float(accuracy_score(y_test,y_test_pred))
        return test_acc
