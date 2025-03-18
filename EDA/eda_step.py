import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import sys
import subprocess

try:
    import imblearn
except ImportError:
    print("imbalanced-learn not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
    import imblearn  # try importing again
    
from imblearn.over_sampling import SMOTE
from typing import Dict, List


# noshow_data_df DataFrame data preprocessing step
def ml_eda_step(
    noshow_data_df: pd.DataFrame,
    target_col: str,
    num_map_dict: Dict,
    standard_list: List,
    one_hot_list: List,
    model_test_size: float,
    model_random_state: int,
):
    ## Drop rows with empty cells
    fil_noshow_data_df = noshow_data_df.dropna()

    ## Drop duplicates in 'booking_id' columns
    fil_noshow_data_df = fil_noshow_data_df.drop_duplicates(
        subset="booking_id", keep="first"
    )

    ## Drop 'booking_id' column
    fil_noshow_data_df = fil_noshow_data_df.drop(columns="booking_id")

    ## Perform manual conversion for price and num_adults
    ### Convert USD to SGD and drop currency tag -> Output is new column named 'sgd_price'
    fil_noshow_data_df["sgd_price"] = fil_noshow_data_df["price"].apply(convert_price)

    ### Drop 'price' column
    fil_noshow_data_df = fil_noshow_data_df.drop(columns="price")

    ### Apply num_map_dict to 'num_adults' column
    fil_noshow_data_df["num_adults"] = fil_noshow_data_df["num_adults"].replace(
        num_map_dict
    )

    ### Perform label categorization to make target columns has categorical values
    lab_enc = LabelEncoder()
    fil_noshow_data_df[target_col] = lab_enc.fit_transform(
        fil_noshow_data_df[target_col]
    )

    ### Perform standardization and one-hot encoding
    #### Standardization [to z-scale]
    scaler = StandardScaler()

    fil_noshow_data_df[standard_list] = scaler.fit_transform(
        fil_noshow_data_df[standard_list]
    )

    #### One-hot encoding
    encoded_noshow_data_df = pd.get_dummies(
        fil_noshow_data_df, columns=one_hot_list, drop_first=False
    )
    bool_col = encoded_noshow_data_df.select_dtypes(include=["bool"]).columns
    encoded_noshow_data_df[bool_col] = encoded_noshow_data_df[bool_col].astype(int)

    ## Prepare model testing and training dataset for both features and target
    ## Prepare numerical data preprocessing
    target_data_df = encoded_noshow_data_df[target_col]
    feature_data_df = encoded_noshow_data_df.drop(columns=target_col)
    preprocessor, X_train, X_test, Y_train, Y_test = model_data_prep(
        target_data_df, feature_data_df, model_test_size, model_random_state
    )

    return encoded_noshow_data_df, preprocessor, X_train, X_test, Y_train, Y_test


# Currency conversion
# def convert_price(price: str):
#     ## Check if price is in USD/SGD
#     if "USD$" in price:
#         value = float(price.replace("USD$ ", "").strip())
#         ### Convert USD to SGD by multiplying by 1.34
#         value = value * 1.34
#     else:
#         # Extract number after SGD$ and convert to float
#         value = float(price.replace("SGD$ ", "").strip())
#     return value


# Prepare data for model use in later steps
def model_data_prep(
    target_data_df: pd.DataFrame,
    feature_data_df: pd.DataFrame,
    model_test_size: float,
    model_random_state: int,
):
    ## Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        feature_data_df,
        target_data_df,
        test_size=model_test_size,
        random_state=model_random_state,
    )

    ## Perform SMOTE to balance dataset
    smote = SMOTE(sampling_strategy="auto", random_state=model_random_state)
    resampled_X_train, resampled_Y_train = smote.fit_resample(X_train, Y_train)

    ## Preprocessing for numerical data
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_data_df.columns)]
    )

    return preprocessor, resampled_X_train, X_test, resampled_Y_train, Y_test