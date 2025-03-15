import yaml


def setup_stage():
    # Load configuration file
    with open("/content/drive/MyDrive/HotelNoShowPrediction/task_2/src/cfg/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Accessing configuration settings
    db_path = config["database"]["path"]
    target_col = config["features"]["target"]
    num_map_dict = config["features"]["num_map"]
    standard_list = config["features"]["standard_list"]
    one_hot_list = config["features"]["one_hot_list"]
    model_test_size = config["model"]["test_size"]
    model_random_state = config["model"]["random_state"]
    model_name_list = config["model"]["name_list"]
    model_search_method = config["model"]["search_method"]
    model_cv_num = config["model"]["cv_num"]
    model_scoring = config["model"]["scoring"]
    model_num_iter = config["model"]["num_iter"]
    model_num_jobs = config["model"]["num_jobs"]

    model_param_dict = {}

    if "Logistic Regression" in model_name_list:
        model_param_dict["Logistic Regression"] = {
            "model__solver": config["logistic"]["solver_list"],
            "model__max_iter": config["logistic"]["max_iter_list"],
            "model__C": config["logistic"]["c_list"],
            "model__class_weight": config["logistic"]["class_weight_list"],
        }

    if "Random Forest" in model_name_list:
        model_param_dict["Random Forest"] = {
            "model__n_estimators": config["rand_forest"]["est_list"],
            "model__max_depth": config["rand_forest"]["depth_list"],
            "model__class_weight": config["logistic"]["class_weight_list"],
        }

    if "SVC" in model_name_list:
        model_param_dict["SVC"] = {
            "model__C": config["svc"]["c_list"],
            "model__kernel": config["svc"]["kernel_list"],
            "model__class_weight": config["logistic"]["class_weight_list"],
        }

    if "MLP" in model_name_list:
        model_param_dict["MLP"] = {
            "model__hidden_layer_sizes": config["mlp"]["hidden_layer_sizes_list"],
            "model__activation": config["mlp"]["activation_list"],
            "model__solver": config["mlp"]["solver_list"],
            "model__learning_rate": config["mlp"]["learning_rate_list"],
            "model__max_iter": config["mlp"]["max_iter_list"],
        }

    if "Naive Bayes" in model_name_list:  # Check on parameter list for Naive Bayes
        model_param_dict["Naive Bayes"] = {
            "model__alpha": config["bayes"]["alpha_list"],
        }

    if "XG Boost" in model_name_list:
        model_param_dict["XG Boost"] = {
            "model__learning_rate": config["xgb"]["learning_rate_list"],
            "model__max_depth": config["xgb"]["max_depth_list"],
            "model__subsample": config["xgb"]["subsample_list"],
        }

    return (
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
    )
