import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cv2

class data_A1:
    def train(self):    #get the training dataset
        sum = 5000

        img = list(range(0,sum))
        for i in range(0,sum):
            img_ = cv2.imread('././datasets/celeba/img/%d.jpg' % i)     #Read the .jpg pictures
            img_gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)    #Convert to the gray value
            img_gray_re = img_gray[20:158,20:198]   #Crop image
            img[i] = img_gray_re.flatten()
        X_raw = np.array(img)

        labels = pd.read_csv('././datasets/celeba/labels.csv')  #Read the labels
        Y_raw = labels['gender']
        Y = np.array(Y_raw.values).flatten()[0:sum]

        scaler = StandardScaler()   #Standadize the data
        scaler = scaler.fit(X_raw)
        X_sta = scaler.transform(X_raw)
        self.scaler = scaler

        n_componets = 120   #Use PCA to reduce the demensianlity of input data
        pca = PCA(n_components=n_componets,random_state=99)
        pca.fit(X_sta)
        print('A1_pca variance',pca.explained_variance_ratio_.sum())    #Check the information kept after dimensionality reduction
        X = pca.transform(X_sta)
        self.pca = pca
        return X, Y

    def test(self): #get the testing dataset
        sum = 1000

        img = list(range(0,sum))
        for i in range(0,sum):
            img_ = cv2.imread('././datasets/celeba_test/img/%d.jpg' % i)    #Read the .jpg pictures
            img_gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)    #Convert to the gray value
            img_gray_re = img_gray[20:158,20:198]   #Crop image
            img[i] = img_gray_re.flatten()
        X_raw = np.array(img)

        labels = pd.read_csv('././datasets/celeba_test/labels.csv')      #Read the labels
        Y_raw = labels['gender']
        Y = np.array(Y_raw.values).flatten()[0:sum]

        X_sta = self.scaler.transform(X_raw)    #Standadize the data
        X = self.pca.transform(X_sta)   #Use PCA to reduce the demensianlity of input testing data
        return X, Y
