# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 23:52:40 2016

@author: Digbalay Bose
"""


import numpy as np
import pandas as pd

bike_data=pd.read_csv('bikeDataTrainingUpload.csv')
target="cnt"
cols_to_drop=["casual","registered"]
cols_to_encode=["season","mnth","weekday","weathersit"]
bike_data.drop(cols_to_drop, axis=1, inplace=True)

#dummy encoding of the training data
dummies=[]
bike_dat=bike_data
for col in cols_to_encode:
    dummies.append(pd.get_dummies(bike_data[col]))
set_dummies = pd.concat(dummies, axis=1)    
bike_data=pd.concat((bike_data,set_dummies),axis=1)
bike_data=bike_data.drop(cols_to_encode,axis=1)

#cross validation split
from sklearn import cross_validation
X=bike_data
total_train=X
y=bike_data['cnt'].values
train,test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.20,random_state=0)
colsdr=['cnt']
total_train.drop(colsdr,axis=1,inplace=True)
train.drop(colsdr,axis=1,inplace=True)
test.drop(colsdr,axis=1,inplace=True)
X_tr=train.as_matrix()


from sklearn.metrics import mean_squared_error

#Linear Regression
from sklearn.linear_model import LinearRegression
model_1=LinearRegression()
model_1.fit(train,y_train)
y_linear_regression_predict=model_1.predict(test) #on cross validation set
mn_sq_err_lr=mean_squared_error(y_test,y_linear_regression_predict)
rms_err_lr=np.sqrt(mn_sq_err_lr)
print('\n Error due to Linear regression:')
print(rms_err_lr)

#Ridge regression
from sklearn.linear_model import Ridge
model_2=Ridge(0.5)
model_2.fit(train,y_train)
y_ridge=model_2.predict(test) #on cross validation set
mn_sq_err_ridge=mean_squared_error(y_test,y_ridge)
rms_err_ridge=np.sqrt(mn_sq_err_ridge)
print('\n Error due to Ridge regression:')
print(rms_err_ridge)

#Lasso
from sklearn import linear_model
lasso_model=linear_model.Lasso(alpha=2.5)
lasso_model.fit(X_tr,y_train)
y_lasso=lasso_model.predict(test)
mn_sq_err_lasso=mean_squared_error(y_test,y_lasso)
rms_err_lasso=np.sqrt(mn_sq_err_lasso)
print('\n Error due to Lasso:')
print(rms_err_lasso)

from sklearn.svm import SVR
#SVR on the split train and test set
clf_cross=SVR(C=61900, epsilon=82, kernel='rbf')
clf_cross.fit(X_tr,y_train)
test_ar=test.as_matrix()
y_predict_SVR=clf_cross.predict(test_ar) #oncross validation set
mn_sq_err_SVR=mean_squared_error(y_test,y_predict_SVR)
rms_err_SVR=np.sqrt(mn_sq_err_SVR)
print('\n Error due to SVR:')
print(rms_err_SVR)


clf_tot=SVR(C=61900,epsilon=82,kernel='rbf')
X_tot=total_train.as_matrix()
clf_tot.fit(X_tot,y)



print('\n Testing phase')
test_bike_data=pd.read_csv('TestX.csv')
#encoding phase same as training data
dummies_test=[]
for col_index in cols_to_encode:
    dummies_test.append(pd.get_dummies(test_bike_data[col_index]))
set_dummies_test = pd.concat(dummies_test, axis=1)    
test_bike_data=pd.concat((test_bike_data,set_dummies_test),axis=1)
test_bike_data=test_bike_data.drop(cols_to_encode,axis=1)
test_bike_ar=test_bike_data.as_matrix()



#prediction of the test data using the model learned by total training data
predict_test_SVR_tot=clf_tot.predict(test_bike_ar)
predict_test_SVR=np.absolute(predict_test_SVR_tot)
df_op=pd.DataFrame(predict_test_SVR)
df_op.to_csv('143070026.csv')
