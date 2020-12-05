### quick xgb_test
# this scripts load the train test set
# trains a stock xgb
# 

# Library import
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Load data
train = pd.read_csv("data/train.csv")
validation = pd.read_csv("data/test.csv")

target = train['target']
ft_train = train.drop(['target', 'ID_code'], axis=1)

# basic feature engineering
# 
train_scaler = StandardScaler() 
X_scale = train_scaler.fit_transform(ft_train)
X_train, X_test, y_train, y_test = \
        train_test_split(X_scale, target, test_size=.4, 
                         random_state=42)


## Instantiating xgboost
param = {}
param['gpu_id'] = 1
param['tree_method'] = 'gpu_hist'
model = XGBClassifier(**param)

## fitting based on auc metric
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          eval_metric='auc',
          verbose=True)

# scale the validation set to match the scaling done on the train
sub_ft_test = validation.drop(['ID_code'], axis=1)
X_submit_scale = train_scaler.transform(sub_ft_test)
pred_results = model.predict(X_submit_scale)

# spit out the train and test aucs to see then we can compare to leaderboard
# we use pred_prob col 1 as we need the probabilities for the highest valued class
# for the roc function
y_pred_prob = model.predict_proba(X_test)
y_train_pred_prob = model.predict_proba(X_train)

test_roc = roc_auc_score(y_test, y_pred_prob[:,1])

train_roc = roc_auc_score(y_train, y_train_pred_prob[:,1])

# create submit frame to put into kaggle
filename = 'submit_test_tr-{:.3f}_ts-{:.3f}.csv'.format(train_roc, test_roc)
#print(filename)
submit_frame = validation['ID_code'].copy().to_frame()
submit_frame['target'] = pred_results 
submit_frame.to_csv(filename, index=False)