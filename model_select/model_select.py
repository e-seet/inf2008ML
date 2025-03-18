import pandas as pd
import time
import os
import sys 
import importlib

# os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# if os.path.exists(file_path):
#     print("✅ File found:", file_path)
# else:
#     print("❌ File not found. Check the path.")

sys.path.append("/content/drive/MyDrive/HotelNoShowPrediction/task_2/src/setup")

import setup as setup
importlib.reload(setup)  # Reload the module to reflect new changes

import duration_cal as duration_cal

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from typing import Dict


# Pre-select a few models then execute model training and optimization
# To get the best parameters for model evaluation in the next step
def model_selection(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    model_random_state: int,
    model_search_method: str,
    model_cv_num: int,
    model_scoring: str,
    model_num_iter: int,
    model_num_jobs: int,
    model_param_dict: Dict,
):
    ## Define models and hyper-parameters
    model_dict = {}

    if "Logistic Regression" in model_param_dict:
        model_dict["Logistic Regression"] = {
            "model": LogisticRegression(random_state=model_random_state),
            "params": model_param_dict["Logistic Regression"],
        }
    if "Random Forest" in model_param_dict:
        model_dict["Random Forest"] = {
            "model": RandomForestClassifier(random_state=model_random_state),
            "params": model_param_dict["Random Forest"],
        }
    if "SVC" in model_param_dict:  # Support Vector Classifier
        model_dict["SVC"] = {"model": SVC(), "params": model_param_dict["SVC"]}
    if "MLP" in model_param_dict:
        model_dict["MLP"] = {
            "model": MLPClassifier(random_state=model_random_state),
            "params": model_param_dict["MLP"],
        }
    if "Naive Bayes" in model_param_dict:  # TODO: Not done
        model_dict["Naive Bayes"] = {
            "model": BernoulliNB(),
            "params": model_param_dict["Naive Bayes"],
        }
    if "XG Boost" in model_param_dict:
        model_dict["XG Boost"] = {
            "model": XGBClassifier(
                objective="reg:squarederror", random_state=model_random_state
            ),
            "params": model_param_dict["XG Boost"],
        }

    ## Initialize empty dictionary to store best models
    best_estimators_dict = {}

    ## Loop through each model
    for model_name, mp in model_dict.items():
        model_start_time = time.time()
        print(f"Processing {model_name} now...")
        ### Create pipeline with preprocessing and model
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", mp["model"])]
        )

        if model_search_method == "grid":
            #### Use GridSearchCV for hyper-parameter tuning
            search = GridSearchCV(
                pipeline,
                param_grid=mp["params"],
                cv=model_cv_num,
                scoring=model_scoring,
                n_jobs=model_num_jobs,
            )
        elif model_search_method == "random":
            #### Use RandomizedSearchCV for hyper-parameter tuning
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=mp["params"],
                n_iter=model_num_iter,
                cv=model_cv_num,
                scoring=model_scoring,
                random_state=model_random_state,
                n_jobs=model_num_jobs,
            )

        search.fit(X_train, Y_train)

        ### Save best model and use parameters for model evaluation
        best_estimators_dict[model_name] = search.best_estimator_
        print(f"Best parameters for {model_name}: {search.best_params_}")

        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        model_duration, model_tag = duration_cal.duration_cal(model_total_time)
        print(f"{model_name} has run tuning for {model_duration:.3f} {model_tag}!")
        print()

    return best_estimators_dict