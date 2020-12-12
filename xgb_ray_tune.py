import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from ray import tune
import pandas as pd

def train_santander(config):
    # Load dataset
    #data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)

    train = pd.read_csv("/home/jovyan/work/ray_experiments/data/train.csv")
    
    data = train.drop(['target', 'ID_code'], axis=1)
    labels = train['target']

    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    # Train the classifier
    results = {}
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        evals_result=results,
        verbose_eval=False)
    # Return prediction accuracy
    accuracy = 1. - results["eval"]["error"][-1]
    tune.report(mean_accuracy=accuracy, done=True)

if __name__ == '__main__':

    config = {
         "objective": "binary:logistic",
         "eval_metric": ["logloss", "error"],
         "max_depth": tune.randint(1, 9),
         "min_child_weight": tune.choice([1, 2, 3]),
         "subsample": tune.uniform(0.5, 1.0),
         "eta": tune.loguniform(1e-4, 1e-1)
    }

    analysis = tune.run(
         train_santander,
         resources_per_trial={"cpu": 1},
         config=config,
         num_samples=10)