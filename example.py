

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import utils as ut
import TPD_loss as tpd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



#LOAD DATA
#----------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

#Select the privileged feature for the example
pi_features = ut.feat_correlation(X,y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

SS = StandardScaler()
X_train = pd.DataFrame(SS.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(SS.transform(X_test), columns = X_train.columns)

# Get the privileged feature
pri = X_train[pi_features]
pri_test = X_test[pi_features]

#Drop the privileged feature from the train set
X_trainr = X_train.drop(pi_features, axis = 1)
X_testr = X_test.drop(pi_features, axis = 1)

            
#TEACHER (REGULAR + PRIV)
#----------------------------------------------------------
model =  ut.nn_binary_clasification( X_train.shape[1], [])     
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X_train, y_train, epochs=300, batch_size=128, verbose = 0, validation_split = 0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5)])

#Measure test error
y_pre_up = np.ravel([np.round(i) for i in model.predict(X_test)])
y_proba_tr = model.predict(X_train)
err_teacher = 1-accuracy_score(y_test, y_pre_up)

            
          
#### TPD
#### ---------------------------------------------------------- 
delta_i = np.array((y_train == np.round(np.ravel(y_proba_tr)))*1)
yy_TPD = np.column_stack([np.ravel(y_train), np.ravel(y_proba_tr), delta_i])


model =  ut.nn_binary_clasification( X_trainr.shape[1], [])     
model.compile(loss= tpd.loss_TPD(1, 1, 0.5), optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X_trainr, yy_TPD, epochs=300, batch_size=128, verbose = 0, validation_split = 0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5)])     


#Measure test error
y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
err_tpd = 1-accuracy_score(y_test, y_pre)


# %%
