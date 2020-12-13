## Feature Engineering Script
# this will read in the data and create the output needed 
import pandas as pd
import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

logging.info("Loading Data")

train = pd.read_csv("data/train.csv")
private_test = pd.read_csv("data/test.csv")
var_columns = train.columns.tolist()[2:]

# for this dataset distinctness
# and counts are important
# xgb models also aren't good at look at frequencies of data

logging.info("Processing Columns")

for var in tqdm(var_columns):
    train['count_'+var] = train[var].map(train[var].value_counts())
    train['in_test_'+var] = train[var].isin(private_test[var]).astype(int)
    private_test['count_'+var] = private_test[var].map(private_test[var].value_counts())
    private_test['in_test_'+var] = 1 

### save out the frame
logging.info("Writing Out Data")
feat_name = 'feat_1_{0}'.format('baseline')
train.to_feather('data/train' + feat_name + '.feather')
private_test.to_feather('data/test' + feat_name + '.feather')

