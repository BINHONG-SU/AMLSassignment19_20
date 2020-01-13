import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cv2

class data_B2:
    def train(self):     #get the training dataset
        sum = 10000
        img = list(range(0,sum))
        for i in range(0,sum):
            img_ = cv2.imread('././datasets/cartoon_set/img/%d.png' % i)     #Read the  RGB values of evry .png picture
            img_1=img_[100:400,150:350] #Crop image, keep the upper part
            img_2 = cv2.resize(img_1,(75,50))   #resize the image
            img[i] = img_2.flatten()
        X_raw = np.array(img)

        labels = pd.read_csv('././datasets/cartoon_set/labels.csv') #Read the labels
        Y_raw = labels['eye_color']
        Y = np.array(Y_raw.values).flatten()

        X_train, X_val, Y_train, Y_val = train_test_split(X_raw, Y_raw, train_size= 0.8, random_state=0)#split the training set and validation set

        scaler = StandardScaler()   #Standadize the data
        sclaer = scaler.fit(X_train)
        X_train_sta = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        self.scaler = scaler

        n_componets = 305   #Use PCA to reduce the demensianlity of input data
        pca = PCA(n_components=n_componets,random_state=99)
        pca.fit(X_train_sta)
        print('B2_pca variance',pca.explained_variance_ratio_.sum())    #Check the information kept after dimensionality reduction
        print(' ')
        X_train = pca.transform(X_train_sta)
        X_val = pca.transform(X_val)
        self.pca = pca

        return X_train, X_val, Y_train, Y_val

    def test(self): #get the testing dataset
        sum = 2500

        img = list(range(0,sum))
        for i in range(0,sum):
            img_ = cv2.imread('././datasets/cartoon_set_test/img/%d.png' % i)   #Read the  RGB values of evry .png picture
            img_1=img_[100:400,150:350]     #Crop image, keep the upper part
            img_2 = cv2.resize(img_1,(75,50))   #resize the image
            img[i] = img_2.flatten()
        X_raw = np.array(img)

        labels = pd.read_csv('././datasets/cartoon_set_test/labels.csv')    #Read the labels
        Y_raw = labels['eye_color']
        Y_test = np.array(Y_raw.values).flatten()

        X_test_sta = self.scaler.transform(X_raw)   #Standadize the data
        X_test = self.pca.transform(X_test_sta) #Use PCA to reduce the demensianlity of input data

        return X_test, Y_test






























def get_data():
    sum = 10000

    img = list(range(0,sum))
    for i in range(0,sum):
        img_ = cv2.imread('././datasets/cartoon_set/img/%d.png' % i)
        img_1=img_[100:400,150:350]
        img_2 = cv2.resize(img_1,(75,50))
        img[i] = img_2.flatten()
    X = np.array(img)

    labels = pd.read_csv('././datasets/cartoon_set/labels.csv')
    Y_ = labels['eye_color']
    Y = np.array(Y_.values).flatten()#[0:sum]


    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size= 0.6, random_state=0)
    X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, train_size= 0.5, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    n_componets = 300
    pca = PCA(n_components=n_componets,random_state=2)
    pca.fit(X_train)
    print('B2_pca variance',pca.explained_variance_ratio_.sum())
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    X_val = pca.transform(X_val)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val
