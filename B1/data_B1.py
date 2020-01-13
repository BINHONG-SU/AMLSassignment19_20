import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cv2

class data_B1:
    def train(self):     #get the training dataset
        sum = 10000

        img = list(range(0,sum))
        for i in range(0,sum):
            img_ = cv2.imread('././datasets/cartoon_set/img/%d.png' % i)    #Read the .png pictures
            img_1 = img_[50:450,50:450] #Crop image
            img_2 = cv2.resize(img_1,(100,100)) #resize the image
            img_3 = cv2.cvtColor(img_2,cv2.COLOR_RGB2GRAY)  #Convert to the gray value
            img[i] = img_3.flatten()
        X_raw = np.array(img)

        labels = pd.read_csv('././datasets/cartoon_set/labels.csv') #Read the labels
        Y_raw = labels['face_shape']
        Y = np.array(Y_raw.values).flatten()

        X_train, X_val, Y_train, Y_val = train_test_split(X_raw, Y_raw, train_size= 0.8, random_state=0)#split the training set and validation set

        scaler = StandardScaler()   #Standadize the data
        sclaer = scaler.fit(X_train)
        X_train_sta = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        self.scaler = scaler

        n_componets = 200   #Use PCA to reduce the demensianlity of input data
        pca = PCA(n_components=n_componets,random_state=1)
        pca.fit(X_train_sta)
        print('B1_pca variance',pca.explained_variance_ratio_.sum())    #Check the information kept after dimensionality reduction
        X_train = pca.transform(X_train_sta)
        X_val = pca.transform(X_val)
        self.pca = pca

        return X_train, X_val, Y_train, Y_val

    def test(self):
        sum = 2500

        img = list(range(0,sum))
        for i in range(0,sum):
            img_ = cv2.imread('././datasets/cartoon_set_test/img/%d.png' % i)   #Read the .png pictures
            img_1 = img_[50:450,50:450] #Crop image
            img_2 = cv2.resize(img_1,(100,100)) #resize the image
            img_3 = cv2.cvtColor(img_2,cv2.COLOR_RGB2GRAY)  #Convert to the gray value
            img[i] = img_3.flatten()
            X_raw = np.array(img)

            labels = pd.read_csv('././datasets/cartoon_set_test/labels.csv')    #Read the labels
        Y_raw = labels['face_shape']
        Y_test = np.array(Y_raw.values).flatten()

        X_test_sta = self.scaler.transform(X_raw)    #Standadize the data
        X_test = self.pca.transform(X_test_sta) #Use PCA to reduce the demensianlity of input data

        return X_test, Y_test
