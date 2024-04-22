import pandas as pd
from NSGA.Logit import fit_logit
from DataBalancer import balance_data
import os
import logging
logging.basicConfig(level="INFO")


def generate_control_group():
    """
    generate models without optimization, but with 4 different sampling techniques as control group
    :return: control_group (dict)
    """
    # if there is already the result, then no need to run with current balancing strategy
    filename = "./Experiment_result/F1-Score ControlGroup.txt"
    if os.path.exists(filename):
        logging.info(f"The file '{filename}' already exists, no need to generate control group!")
        return

    # If no experiment result, generate control group
    # read x_validation and y_validation
    x_valid = pd.read_csv("./Datasets/x_valid.csv", index_col=0)
    y_valid = pd.read_csv("./Datasets/y_valid.csv", index_col=0)

    # generate modified x_train and y_train
    sampling_list = ["none", "under", "over", "smote"]
    control_group = {}
    for strategy in sampling_list:
        x_train, y_train = balance_data(sampling=strategy, random_seed=343)

        logit_result, score_dict, confusion_matrix = fit_logit(x_train=x_train, x_valid=x_valid,
                                                               y_train=y_train, y_valid=y_valid)

        # get the recall value
        f1_score_error = round(1 - score_dict["f1"], 4)
        control_group[strategy] = f1_score_error

        # save the model detail as txt file
        original_model = str(logit_result.summary())
        with open("./Experiment_result/Model ControlGroup SamplingStrategy " + str(strategy) + ".txt", "w") as file:
            file.write(original_model)

    # save the recall errors as txt file
    with open("./Experiment_result/F1-Score Error ControlGroup.txt", "w") as file:
        for key, value in control_group.items():
            file.write(str(key) + ": " + str(value) + "\n")

    logging.info("Control group generated and saved!")

