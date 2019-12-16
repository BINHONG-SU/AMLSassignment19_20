#import lab2_landmarks as l2
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
sys.path.append('./B1/')
from B1 import B1
sys.path.append('./B2/')
from B2 import B2
# ======================================================================================================================
# Data preprocessing
#X, y = l2.extract_features_labels()
#Y = np.array([y, -(y - 1)]).T
sum = 100
imageA_0 = cv2.imread('./datasetsA/img/0.jpg')
imageA_0_gray = cv2.cvtColor(imageA_0, cv2.COLOR_BGR2GRAY)
imageA_1 = cv2.imread('./datasetsA/img/1.jpg')
imageA_1_gray = cv2.cvtColor(imageA_1, cv2.COLOR_BGR2GRAY)
facesA = np.array([imageA_0_gray, imageA_1_gray])
for i in range(2,sum):
#    image[i] = cv2.imread('./datasetsA/img/%d.jpg' % i)
    imageA_i = cv2.imread('./datasetsA/img/%d.jpg' % i)
    imageA_i_gray = cv2.cvtColor(imageA_i,cv2.COLOR_RGB2GRAY)
    facesA_new = np.append(facesA, imageA_i_gray)
    dim = facesA.shape
    facesA = facesA_new.reshape(dim[0]+1,dim[1],dim[2])
dataset = pd.read_csv('./datasetsA/labels.csv')
y = dataset[['gender']]
Y = np.array(y.values).flatten()[0:sum]
X = facesA.reshape(sum, dim[1]*dim[2])
X_train_A1, X_temp_A1, Y_train_A1, Y_temp_A1 = train_test_split(X, Y, train_size= 0.6, random_state=0)
X_test_A1, X_val_A1, Y_test_A1, Y_val_A1 = train_test_split(X_temp_A1, Y_temp_A1, train_size = 0.5, random_state=0)
print(X_test_A1.shape)
#data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
#
#acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
#Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
#model_A2 = A2(args...)
#acc_A2_train = model_A2.train(args...)
#acc_A2_test = model_A2.test(args...)
#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
#model_B1 = B1(args...)
#acc_B1_train = model_B1.train(args...)
#acc_B1_test = model_B1.test(args...)
#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
#model_B2 = B2(args...)
#acc_B2_train = model_B2.train(args...)
#acc_B2_test = model_B2.test(args...)
#Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
#print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                        acc_A2_train, acc_A2_test,
#                                                        acc_B1_train, acc_B1_test,
#                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'
