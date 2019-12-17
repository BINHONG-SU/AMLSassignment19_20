import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cv2

sum = 10000

img_B1 = list(range(0,sum))
img_B2 = list(range(0,sum))
for i in range(0,sum):
    img_i = cv2.imread('././datasets/cartoon_set/img/%d.png' % i)

    img1_1 = img_i[50:450,50:450]
    img1_2 = cv2.resize(img1_1,(100,100))
    img1_3 = cv2.cvtColor(img1_2,cv2.COLOR_RGB2GRAY)
    img_B1[i] = img1_3.flatten()


    img2_1=img_i[100:400,150:350]
    img2_2 = cv2.resize(img2_1,(75,50))
    img_B2[i] = img2_2.flatten()

X_B1 = np.array(img_B1)
X_B2 = np.array(img_B2)


labels_B = pd.read_csv('././datasets/cartoon_set/labels.csv')
Y_B1_ = labels_B['face_shape']
Y_B1 = np.array(Y_B1_.values).flatten()#[0:sum]
Y_B2_ = labels_B['eye_color']
Y_B2 = np.array(Y_B2_.values).flatten()#[0:sum]


def B1():
    X_train_B1, X_temp_B1, Y_train_B1, Y_temp_B1 = train_test_split(X_B1, Y_B1, train_size= 0.6, random_state=0)
    X_test_B1, X_val_B1, Y_test_B1, Y_val_B1 = train_test_split(X_temp_B1, Y_temp_B1, train_size= 0.5, random_state=0)
    scaler = StandardScaler()
    X_train_B1 = scaler.fit_transform(X_train_B1)
    X_test_B1 = scaler.transform(X_test_B1)
    X_val_B1 = scaler.transform(X_val_B1)

    n_componets = 200
    pca = PCA(n_components=n_componets)
    pca.fit(X_train_B1)
    print('B1_pca variance',pca.explained_variance_ratio_.sum())
    X_train_B1 = pca.transform(X_train_B1)
    X_test_B1 = pca.transform(X_test_B1)
    X_val_B1 = pca.transform(X_val_B1)

    X_train_B1 = scaler.fit_transform(X_train_B1)
    X_test_B1 = scaler.transform(X_test_B1)
    X_val_B1 = scaler.transform(X_val_B1)
    return X_train_B1, X_test_B1, X_val_B1, Y_train_B1, Y_test_B1, Y_val_B1

def B2():
    X_train_B2, X_temp_B2, Y_train_B2, Y_temp_B2 = train_test_split(X_B2, Y_B2, train_size= 0.6, random_state=0)
    X_test_B2, X_val_B2, Y_test_B2, Y_val_B2 = train_test_split(X_temp_B2, Y_temp_B2, train_size= 0.5, random_state=0)
    scaler = StandardScaler()
    X_train_B2 = scaler.fit_transform(X_train_B2)
    X_test_B2 = scaler.transform(X_test_B2)
    X_val_B2 = scaler.transform(X_val_B2)

    n_componets = 300
    pca = PCA(n_components=n_componets)
    pca.fit(X_train_B2)
    print('B2_pca variance',pca.explained_variance_ratio_.sum())
    X_train_B2 = pca.transform(X_train_B2)
    X_test_B2 = pca.transform(X_test_B2)
    X_val_B2 = pca.transform(X_val_B2)

    X_train_B2 = scaler.fit_transform(X_train_B2)
    X_test_B2 = scaler.transform(X_test_B2)
    X_val_B2 = scaler.transform(X_val_B2)

    return X_train_B2, X_test_B2, X_val_B2, Y_train_B2, Y_test_B2, Y_val_B2
