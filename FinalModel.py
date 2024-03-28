import pandas as pd
import numpy as np
from DataBalancer import balance_data
from imblearn.over_sampling import SMOTE
from NSGA.Logit import fit_logit
import logging
logging.basicConfig(level="INFO")


def build_final_model(random_seed=343):
    """
    modeling with the best balancing technique and best feature combination
    :return:
    """
    # 1. prepare the dataset
    x_train = pd.read_csv("./Datasets/x_train.csv", index_col=0)
    y_train = pd.read_csv("./Datasets/y_train.csv", index_col=0)
    x_test = pd.read_csv("./Datasets/x_test.csv", index_col=0)
    y_test = pd.read_csv("./Datasets/y_test.csv", index_col=0)
    best_solution = pd.read_csv("./Experiment_result/BestSolution smote.csv", header=None)
    best_solution = best_solution.iloc[0].tolist()
    best_solution = np.array(best_solution)
    best_features = x_test.columns[best_solution == 1]

    # 2. balance the dataset using smote
    smoter = SMOTE(random_state=random_seed)
    x_train, y_train = smoter.fit_resample(x_train, y_train)

    # 3. fit the model using the selected features
    x_train_selected = x_train.loc[:, best_features]
    x_test_selected = x_test.loc[:, best_features]
    logit_result, score_dict, cm = fit_logit(x_train=x_train_selected, x_valid=x_test_selected, y_train=y_train, y_valid=y_test)
    final_model = str(logit_result.summary())
    with open("./Experiment_result/Model Final SMOTE Summary.txt", "w") as file:
        file.write(final_model + "\n")
        file.write(str(score_dict) + "\n")
        file.write(str(cm) + "\n") 

    logging.info("Final model saved!")


