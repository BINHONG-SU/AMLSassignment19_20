#import lab2_landmarks as l2
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import gc
sys.path.append('./A1/')
import data_A1
from A1 import A1
sys.path.append('./A2/')
from A2 import A2
import data_A2
from sklearn.preprocessing import StandardScaler
sys.path.append('./B1/')
import data_B1
from B1 import B1
sys.path.append('./B2/')
import data_B2
from B2 import B2
#X, Y_gender, Y_smiling = l2.extract_features_labels()
#=================================================================
'''
X_train_A1, X_test_A1,  Y_train_A1, Y_test_A1  = data_A1.get_data()
X_train_A2, X_test_A2,  Y_train_A2, Y_test_A2 = data_A2.get_data()
X_train_B1, X_test_B1, X_val_B1, Y_train_B1, Y_test_B1, Y_val_B1 = data_B1.get_data()
X_train_B2, X_test_B2, X_val_B2, Y_train_B2, Y_test_B2, Y_val_B2 = data_B2.get_data()
'''
'''
np.save('img_A11',X_train_A1)
np.save('img_A12',X_test_A1)
np.save('img_A21',X_train_A2)
np.save('img_A22',X_test_A2)
np.save('img_A111',Y_train_A1)
np.save('img_A122',Y_test_A1)
np.save('img_A211',Y_train_A2)
np.save('img_A222',Y_test_A2)
'''



X_train_A1=np.load('img_A11.npy')
X_test_A1=np.load('img_A12.npy')
X_train_A2=np.load('img_A21.npy')
X_test_A2=np.load('img_A22.npy')
Y_train_A1=np.load('img_A111.npy')
Y_test_A1=np.load('img_A122.npy')
Y_train_A2=np.load('img_A211.npy')
Y_test_A2=np.load('img_A222.npy')







'''
 ############################LAND MARK#####################################
#np.save('img_A',X)
X=np.load('img_A.npy')

#np.save('A1_Y_gender',Y_gender)
Y_gender=np.load('A1_Y_gender.npy')

#np.save('A2_Y_smiling',Y_smiling)
Y_smiling=np.load('A2_Y_smiling.npy')

#=================================================================
#X, Y_gender, Y_smiling = shuffle(X,Y_gender,Y_smiling)
#Y = np.array([y, -(y - 1)]).T

dim = X.shape
X_train_A1, X_test_A1, Y_train_A1, Y_test_A1 = train_test_split(X.reshape((dim[0], dim[1]*dim[2])), Y_gender, train_size= 0.8, random_state=0)
scaler = StandardScaler()
X_train_A1 = scaler.fit_transform(X_train_A1)
X_test_A1 = scaler.transform(X_test_A1)
#print(X_test_A1.shape)

X_A2 = X
dim = X_A2.shape
X_train_A2, X_test_A2, Y_train_A2, Y_test_A2 = train_test_split(X_A2.reshape((dim[0], dim[1]*dim[2])), Y_smiling, train_size= 0.8, random_state=1)
X_train_A2 = scaler.fit_transform(X_train_A2)
X_test_A2 = scaler.transform(X_test_A2)
#print(X_test_A2.shape)
'''
#==========================================================================================

model_A1 = A1()
acc_A1_train = model_A1.train(X_train_A1,Y_train_A1)
acc_A1_test = model_A1.test(X_test_A1,Y_test_A1)
print('acc_A1_train',acc_A1_train)
print('acc_A1_test',acc_A1_test)
model_A1.plot_learning_curve(X_train_A1,Y_train_A1)
print(' ')
del X_train_A1, X_test_A1, Y_train_A1, Y_test_A1
gc.collect()
#=========================================================================================
model_A2 = A2()
acc_A2_train = model_A2.train(X_train_A2,Y_train_A2)
acc_A2_test = model_A2.test(X_test_A2,Y_test_A2)
print('acc_A2_train',acc_A2_train)
print('acc_A2_test',acc_A2_test)
model_A2.plot_learning_curve(X_train_A2,Y_train_A2)
print(' ')
del X_train_A2, X_test_A2, Y_train_A2, Y_test_A2
gc.collect()





'''
np.save('img_B11',X_train_B1)
np.save('img_B12',X_test_B1)
np.save('img_B13',X_val_B1)
np.save('img_B21',X_train_B2)
np.save('img_B22',X_test_B2)
np.save('img_B23',X_val_B2)
np.save('img_B111',Y_train_B1)
np.save('img_B122',Y_test_B1)
np.save('img_B133',Y_val_B1)
np.save('img_B211',Y_train_B2)
np.save('img_B222',Y_test_B2)
np.save('img_B233',Y_val_B2)
'''



X_train_B1=np.load('img_B11.npy')
X_test_B1=np.load('img_B12.npy')
X_val_B1=np.load('img_B13.npy')
X_train_B2=np.load('img_B21.npy')
X_test_B2=np.load('img_B22.npy')
X_val_B2=np.load('img_B23.npy')
Y_train_B1=np.load('img_B111.npy')
Y_test_B1=np.load('img_B122.npy')
Y_val_B1=np.load('img_B133.npy')
Y_train_B2=np.load('img_B211.npy')
Y_test_B2=np.load('img_B222.npy')
Y_val_B2=np.load('img_B233.npy')

#========================================================================

model_B1 = B1()
acc_B1_val, acc_B1_train = model_B1.train(X_train_B1, Y_train_B1, X_val_B1, Y_val_B1)
acc_B1_test = model_B1.test(X_test_B1, Y_test_B1)
print(acc_B1_val)
print(acc_B1_train)
print(acc_B1_test)
model_B1.plot_learning_curve(X_train_B1,Y_train_B1)
del X_train_B1, X_test_B1,X_val_B1, Y_train_B1, Y_test_B1, Y_val_B1
gc.collect()


model_B2 = B2()
acc_B2_val, acc_B2_train = model_B2.train(X_train_B2, Y_train_B2, X_val_B2, Y_val_B2)
acc_B2_test = model_B2.test(X_test_B2, Y_test_B2)
print(acc_B2_val)
print(acc_B2_train)
print(acc_B2_test)
model_B2.plot_learning_curve(X_train_B2,Y_train_B2)
del X_train_B2, X_test_B2,X_val_B2, Y_train_B2, Y_test_B2, Y_val_B2
gc.collect()
# ======================================================================================================================
## Print out your results with following format:
#print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                        acc_A2_train, acc_A2_test,
#                                                        acc_B1_train, acc_B1_test,
#                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'
