from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class A1:
    #def __init__(self,para=1):
    #    self.para = para

    def train(self,x_train, y_train):
        param_grid={'C':[0.1,0.5,1,5,10,50,100],'kernel':['linear','rbf']}
        clf = GridSearchCV(SVC(gamma = 'scale'),param_grid,cv = 3)
        clf = clf.fit(x_train, y_train)
        self.clf = clf
        y_train_pred= clf.predict(x_train)
        train_acc = float(accuracy_score(y_train,y_train_pred))
        return train_acc

    def test(self,x_test,y_test):
        y_test_pred = self.clf.predict(x_test)
        test_acc = float(accuracy_score(y_test,y_test_pred))
        return test_acc




'''
#import libraries
#from pandas.plotting import scatter_matrix
#import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import numpy as np
import cv2

#load the data
##sum = 1000
#image = list(range(0,sum))
#gray = list(range(0,sum))
#faces = list(range(0,sum))
##image_0 = cv2.imread('./datasetsA/img/0.jpg')
##image_0_gray = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
##image_1 = cv2.imread('./datasetsA/img/1.jpg')
##image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
##faces = np.array([image_0_gray, image_1_gray])
#img_array = np.asarray(img)
#row0 = image0_gray.shape[0]
#column0 = image0_gray.shape[1]
#faces = np.random.random(size = (sum,row0,column0))
#print(faces.shape)
##for i in range(2,sum):
#    image[i] = cv2.imread('./datasetsA/img/%d.jpg' % i)
##    image_i = cv2.imread('./datasetsA/img/%d.jpg' % i)
##    image_i_gray = cv2.cvtColor(image_i,cv2.COLOR_RGB2GRAY)
##    faces_new = np.append(faces, image_i_gray)
##    dim = faces.shape
##    faces = faces_new.reshape(dim[0]+1,dim[1],dim[2])
##dataset = pd.read_csv('./datasetsA/labels.csv')
##y = dataset[['gender']]
##y = np.array(y.values).flatten()[0:1000]###################################
##x = faces.reshape(sum, dim[1]*dim[2])
# Split the data into training and testing(75% training and 25% testing data)
##x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
##print(x_train.shape)
##scaler = MinMaxScaler() # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
##x_train = scaler.fit_transform(x_train)
##x_test = scaler.transform(x_test)
# sklearn functions implementation
def ModelSelect(x_train,y_train,x_test,y_test):
    # Build Logistic Regression Model
    c_min = 0.1
    c_max = 20
    c_gap = 0.1
    loop = int((c_max - c_min)/c_gap)
    try_para=np.arange(c_min,c_max,c_gap)
    list_acc = list(range(0,loop))
    for i in range(0,loop):
        logreg = LogisticRegression(solver='liblinear',C=try_para[i])
    # Train the model using the training sets
        logreg.fit(x_train, y_train)
        y_pred = logreg.predict(x_test)
        acc = float(accuracy_score(y_test,y_pred))
        list_acc[i] = acc
    #print('Accuracy on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    m=list_acc.index(max(list_acc))
    return (c_min + m*c_gap)

#def ModelIdentified1(x_train,y_train,para):
#    logreg = LogisticRegression(solver='liblinear',C=para)
# Train the model using the training sets
#    logreg.fit(x_train, y_train)
#    y_pred = logreg.predict(x_train)
#    acc = float(accuracy_score(y_train,y_pred))
#    return acc

def ModelIdentified(x_train,y_train,x_test,y_test,para):
    logreg = LogisticRegression(solver='liblinear',C=para)
# Train the model using the training sets
    logreg.fit(x_train, y_train)
    y_pred_train = logreg.predict(x_train)
    y_pred_test = logreg.predict(x_test)
    train_acc = float(accuracy_score(y_train,y_pred_train))
    test_acc = float(accuracy_score(y_test,y_pred_test))
    return train_acc, test_acc
##y_pred = logRegrPredict(x_train, y_train,x_test)
##print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
##print(classification_report(y_test,y_pred))#text report showing the main classification metrics
'''
