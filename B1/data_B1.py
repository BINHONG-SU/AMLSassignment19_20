import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cv2

def get_data():
    sum = 10000

    img = list(range(0,sum))

    for i in range(0,sum):
        img_ = cv2.imread('././datasets/cartoon_set/img/%d.png' % i)

        img_1 = img_[50:450,50:450]
        img_2 = cv2.resize(img_1,(100,100))
        img_3 = cv2.cvtColor(img_2,cv2.COLOR_RGB2GRAY)
        img[i] = img_3.flatten()



    X = np.array(img)


    labels = pd.read_csv('././datasets/cartoon_set/labels.csv')
    Y_ = labels['face_shape']
    Y = np.array(Y_.values).flatten()#[0:sum]


    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size= 0.6, random_state=0)
    X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, train_size= 0.5, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    n_componets = 200
    pca = PCA(n_components=n_componets)
    pca.fit(X_train)
    print('B1_pca variance',pca.explained_variance_ratio_.sum())
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    X_val = pca.transform(X_val)

    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    return X_train, X_test, X_val, Y_train, Y_test, Y_val
