# Import necessary libraries
import sqlite3
import pandas as pd
import time

import setup.setup as setup
import setup.duration_cal as duration_cal
import EDA.eda_step as EDA
import model_select.model_select as model_select
import model_eval.model_eval as model_eval

start_time = time.time()

(
    db_path,
    target_col,
    num_map_dict,
    standard_list,
    one_hot_list,
    model_test_size,
    model_random_state,
    model_search_method,
    model_cv_num,
    model_scoring,
    model_num_iter,
    model_num_jobs,
    model_param_dict,
) = setup.setup_stage()

# Create connection to SQL database
print("1. Connecting to SQL database....")
conn = sqlite3.connect(db_path)
print("Connection done!")

part1_time = time.time()
part1_duration, part1_tag = duration_cal.duration_cal(part1_time - start_time)
print(f"Part 1 has run for {part1_duration:.3f} {part1_tag}!")
print()

# Get data from 'noshow' table
print("2. Extract SQL database table as DataFrame...")
noshow_data_query = "SELECT * FROM noshow;"
noshow_data_df = pd.read_sql_query(noshow_data_query, conn)
print("Extraction done!")

part2_time = time.time()
part2_duration, part2_tag = duration_cal.duration_cal(part2_time - part1_time)
print(f"Part 2 has run for {part2_duration:.3f} {part2_tag}!")
print()

# Using analysis from task_1 EDA, perform data preprocessing, feature data standardization and one-hot encoding
print("3. Performing EDA on DataFrame...")
fil_noshow_data_df, preprocessor, X_train, X_test, Y_train, Y_test = EDA.ml_eda_step(
    noshow_data_df,
    target_col,
    num_map_dict,
    standard_list,
    one_hot_list,
    model_test_size,
    model_random_state,
)
print("EDA done!")

part3_time = time.time()
part3_duration, part3_tag = duration_cal.duration_cal(part3_time - part2_time)
print(f"Part 3 has run for {part3_duration:.3f} {part3_tag}!")
print()

# Pre-select a few models and train models to get best optimized parameters
print("4. Training machine learning models...")
best_estimator_dict = model_select.model_selection(
    preprocessor,
    X_train,
    Y_train,
    model_random_state,
    model_search_method,
    model_cv_num,
    model_scoring,
    model_num_iter,
    model_num_jobs,
    model_param_dict,
)
print("Training done!")

part4_time = time.time()
part4_duration, part4_tag = duration_cal.duration_cal(part4_time - part3_time)
print(f"Part 4 has run for {part4_duration:.3f} {part4_tag}!")
print()

# Evaluate pre-selected models to get mean-squared error and r^2 values to determine which model is better for current dataset
print("5. Evaluating machine learning model...")
model_eval.model_evaluation(X_test, Y_test, best_estimator_dict)
print("Evaluation done!")

part5_time = time.time()
part5_duration, part5_tag = duration_cal.duration_cal(part5_time - part4_time)
print(f"Part 5 has run for {part5_duration:.3f} {part5_tag}!")
print()
print()

# Best model decision -
## If model variance is priority, look for highest R^2
## If predictive accuracy is priority, look for lowerst MSE (0 == Perfect model)

end_time = time.time()
final_time = end_time - start_time
final_duration, final_tag = duration_cal.duration_cal(final_time)

print("Script has reached end of line - It will terminate now!")
print(f"Script has run for {final_duration:.3f} {final_tag}!")
