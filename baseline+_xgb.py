# in the baseline script
# the train test split in the xgb model was high  
# the difference between the actual leaderboard score and the validation set was also high
# we are definitely overfitting

# we need to close up the differences
# between train and validation
# try normalisation
# add a simple gridsearch on xgboost



# Library import
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb

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

dtrain = xgb.DMatrix(X_train, label=y_train)

num_boost_round = 300
params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 1,
    'eta':.1,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'binary:logistic',
}
params['gpu_id'] = 1
params['tree_method'] = 'gpu_hist'

#### Grid Search
# from https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12) # 3-10?
    for min_child_weight in range(5,8) # 1-10?
]

max_auc = float("-Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=10,
        metrics={'auc'},
        early_stopping_rounds=20
    )
    # Update best MAE
    mean_auc = cv_results['test-auc-mean'].max()
    boost_rounds = cv_results['test-auc-mean'].argmax()
    print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))

## Instantiating xgboost
# default lambda = 1 
# default alpha = 0 


xparam = {"reg_alpha": 50, 
        "reg_lambda": 50}
xparam['gpu_id'] = 1
xparam['tree_method'] = 'gpu_hist'
model = XGBClassifier(**xparam)

## fitting based on auc metric
#model.fit(X_train, y_train,
#          eval_set=[(X_train, y_train), (X_test, y_test)],
#          eval_metric='auc',
#          verbose=True)