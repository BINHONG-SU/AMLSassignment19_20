import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cv2

def get_data():
    sum = 5000
    img = list(range(0,sum))
    for i in range(0,sum):
        img_ = cv2.imread('././datasets/celeba/img/%d.jpg' % i)
        img_gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)
        img[i] = img_gray[20:158,20:100].flatten()
    X = np.array(img)
    labels= pd.read_csv('././datasets/celeba/labels.csv')
    Y = labels['smiling']
    Y = np.array(Y.values).flatten()[0:sum]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size= 0.8, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_componets = 100
    pca = PCA(n_components=n_componets)
    pca.fit(X_train)
    print('A2_pca variance',pca.explained_variance_ratio_.sum())
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test
