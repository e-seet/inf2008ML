import pandas as pd
import time
import sys
import importlib
sys.path.append("/content/drive/MyDrive/HotelNoShowPrediction/task_2/src/setup")

import setup as setup
importlib.reload(setup)  # Reload the module to reflect new changes


import duration_cal as duration_cal

from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict


# Evaluate selected models to determine which is more optimized for current dataset
def model_evaluation(
    X_test: pd.DataFrame, Y_test: pd.DataFrame, best_estimator_dict: Dict
):
    ## Initialize dictionary to store results
    eval_result_dict = {}

    for model_name, model in best_estimator_dict.items():
        model_start_time = time.time()
        print(f"Evaluating {model_name} now...")
        ### Predict on test set
        Y_predict = model.predict(X_test)
        ### Calculate evaluation metrics
        confuse_matrix = confusion_matrix(Y_test, Y_predict)
        class_rpt = classification_report(Y_test, Y_predict)
        ### Store results
        eval_result_dict[model_name] = {
            "Confusion Matrix": confuse_matrix,
            "Classification Report": class_rpt,
        }

        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        model_duration, model_tag = duration_cal.duration_cal(model_total_time)
        print(f"{model_name} has run evaluation for {model_duration:.3f} {model_tag}!")
        print()

    ## Info -
    ### Accuracy - Measures overall percentage of correct predictions
    ### Optimal value = 1.0 (100%), Lowest value = 0.0

    ### Precision - Out of instances predicted as positive, how many were actually positive
    ### Optimal value = 1.0 (100%), Lowest value = 0.0

    ### Recall - Out of actual positives, how many were correctly predicted
    ### Optimal value = 1.0 (100%), Lowest value = 0.0

    ### F1-Score - A combined metric that balances precision and recall
    ### Optimal value = 1.0 (100%), Lowest value = 0.0

    ## Display results
    results_df = pd.DataFrame(eval_result_dict).T
    print(results_df)