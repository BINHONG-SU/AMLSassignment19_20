import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cv2
from skimage.feature import hog
sum = 5000

img_A1 = list(range(0,sum))
img_A2 = list(range(0,sum))
#img_A2 = list(range(0,sum))
for i in range(0,sum):
    img = cv2.imread('././datasets/celeba/img/%d.jpg' % i)
    #img_i_400 = img_i[50:450,50:450]
    #img_i_200 = cv2.resize(img_i_400,(100,100))
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_gray_resize1 = img_gray[20:158,20:198]
    img_gray_resize2 = img_gray[20:158,20:100]
    img_A1[i] = hog(img_gray_resize1,orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), feature_vector=True,visualize=False)
    img_A2[i] = hog(img_gray_resize2,orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), feature_vector=True,visualize=False)
    print(img_A1[i].shape)
    print(img_A2[i].shape)
    #dim = img_gray.shape
    #img_resize = cv2.resize(img_gray,(int(dim[0]/4), int(dim[1]/4)))
    #img_A[i] = img_resize.flatten()
X_A1 = np.array(img_A1)
X_A2 = np.array(img_A2)

labels_A = pd.read_csv('././datasets/celeba/labels.csv')
Y_A1 = labels_A['gender']
Y_A1 = np.array(Y_A1.values).flatten()[0:sum]
Y_A2 = labels_A['smiling']
Y_A2 = np.array(Y_A2.values).flatten()[0:sum]


def A1():
    X_train_A1, X_test_A1, Y_train_A1, Y_test_A1 = train_test_split(X_A1, Y_A1, train_size= 0.8, random_state=0)
    scaler = StandardScaler()
    X_train_A1 = scaler.fit_transform(X_train_A1)
    X_test_A1 = scaler.transform(X_test_A1)

    n_componets = 250
    pca = PCA(n_components=n_componets)
    pca.fit(X_train_A1)
    print(pca.explained_variance_ratio_.sum())
    X_train_A1 = pca.transform(X_train_A1)
    X_test_A1 = pca.transform(X_test_A1)

    scaler = StandardScaler()
    X_train_A1 = scaler.fit_transform(X_train_A1)
    X_test_A1 = scaler.transform(X_test_A1)
    return X_train_A1, X_test_A1, Y_train_A1, Y_test_A1


def A2():
    X_train_A2, X_test_A2, Y_train_A2, Y_test_A2 = train_test_split(X_A2, Y_A2, train_size= 0.8, random_state=0)
    scaler = StandardScaler()
    X_train_A2 = scaler.fit_transform(X_train_A2)
    X_test_A2 = scaler.transform(X_test_A2)

    n_componets = 150
    pca = PCA(n_components=n_componets)
    pca.fit(X_train_A2)
    print(pca.explained_variance_ratio_.sum())
    X_train_A2 = pca.transform(X_train_A2)
    X_test_A2 = pca.transform(X_test_A2)

    scaler = StandardScaler()
    X_train_A2 = scaler.fit_transform(X_train_A2)
    X_test_A2 = scaler.transform(X_test_A2)
    return X_train_A2, X_test_A2, Y_train_A2, Y_test_A2
