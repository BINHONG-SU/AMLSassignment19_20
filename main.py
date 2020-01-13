#import lab2_landmarks as l2
import numpy as np
import pandas as pd
import sys
import gc
sys.path.append('./A1/')
from data_A1 import data_A1
from A1 import A1
sys.path.append('./A2/')
from A2 import A2
from data_A2 import data_A2
sys.path.append('./B1/')
from data_B1 import data_B1
from B1 import B1
sys.path.append('./B2/')
from data_B2 import data_B2
from B2 import B2
#=================================================================
# Data preprocessing

data_A1 = data_A1()
X_train_A1, Y_train_A1 = data_A1.train()
X_test_A1, Y_test_A1 = data_A1.test()

data_A2 = data_A2()
X_train_A2, Y_train_A2 = data_A2.train()
X_test_A2, Y_test_A2 = data_A2.test()

data_B1 = data_B1()
X_train_B1, X_val_B1, Y_train_B1, Y_val_B1 = data_B1.train()
X_test_B1, Y_test_B1 = data_B1.test()

data_B2 = data_B2()
X_train_B2, X_val_B2, Y_train_B2, Y_val_B2 = data_B2.train()
X_test_B2, Y_test_B2 = data_B2.test()


#==========================================================================================
# Task A1
model_A1 = A1()
acc_A1_train = model_A1.train(X_train_A1,Y_train_A1)
acc_A1_test = model_A1.test(X_test_A1,Y_test_A1)
print('acc_A1_train',acc_A1_train)
print('acc_A1_test',acc_A1_test)
#model_A1.plot_learning_curve(X_train_A1,Y_train_A1)    #plot the learning curve
print(' ')
del X_train_A1, X_test_A1, Y_train_A1, Y_test_A1        #clean up memory
gc.collect()
#=========================================================================================
# Task A2
model_A2 = A2()
acc_A2_train = model_A2.train(X_train_A2,Y_train_A2)
acc_A2_test = model_A2.test(X_test_A2,Y_test_A2)
print('acc_A2_train',acc_A2_train)
print('acc_A2_test',acc_A2_test)
#model_A2.plot_learning_curve(X_train_A2,Y_train_A2)     #plot the learning curve
print(' ')
del X_train_A2, X_test_A2, Y_train_A2, Y_test_A2    #clean up memory
gc.collect()

#========================================================================
# Task B1
model_B1 = B1()
acc_B1_val, acc_B1_train = model_B1.train(X_train_B1, Y_train_B1, X_val_B1, Y_val_B1)
acc_B1_test = model_B1.test(X_test_B1, Y_test_B1)
print('acc_B1_val',acc_B1_val)
print('acc_B1_train',acc_B1_train)
print('acc_B1_test',acc_B1_test)
print(' ')
#model_B1.plot_learning_curve(X_train_B1,Y_train_B1)    #plot the learning curve
del X_train_B1, X_test_B1,X_val_B1, Y_train_B1, Y_test_B1, Y_val_B1     #clean up memory
gc.collect()

# ======================================================================================================================
# Task B2
model_B2 = B2()
acc_B2_val, acc_B2_train = model_B2.train(X_train_B2, Y_train_B2, X_val_B2, Y_val_B2)
acc_B2_test = model_B2.test(X_test_B2, Y_test_B2)
print('acc_B2_val',acc_B2_val)
print('acc_B2_train',acc_B2_train)
print('acc_B2_test',acc_B2_test)
print(' ')
#model_B2.plot_learning_curve(X_train_B2,Y_train_B2)    #plot the learning curve
del X_train_B2, X_test_B2,X_val_B2, Y_train_B2, Y_test_B2, Y_val_B2     #clean up memory
gc.collect()

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,acc_A2_train, acc_A2_test,acc_B1_train, acc_B1_test,acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'
