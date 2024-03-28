import pandas as pd
import os
from DataBalancer import balance_data
from NSGA.RepeatEvolution import repeat_evolution
import logging
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
logging.basicConfig(level="INFO")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def optimize_by_strategy(total_num_features, sampling_strategy="none", total_repeat_times=10, max_gen_num=10, min_num_features=4):
    """
    Run a full optimization for modeling, given a specific sampling strategy name (none, under, over, smote)
    :param total_num_features:
    :param min_num_features:
    :param max_gen_num:
    :param total_repeat_times:
    :param sampling_strategy:
    :return: recall_list, best_solution, best_recall
    """
    # if there is already the result, then no need to run with current balancing strategy
    filename = "./Experiment_result/Recall Error ComparisonGroup " + str(sampling_strategy) + ".csv"
    if os.path.exists(filename):
        logging.info(f"The file '{filename}' already exists, no need to run optimization with current sampling strategy!")
        return [],[],[]

    # run optimization
    # prepare the datasets
    x_valid = pd.read_csv("./Datasets/x_valid.csv", index_col=0)
    y_valid = pd.read_csv("./Datasets/y_valid.csv", index_col=0)

    x_train, y_train = balance_data(sampling=sampling_strategy, random_seed=343)

    recall_list, best_solution, best_recall = repeat_evolution(total_repeat=total_repeat_times,
                                                               max_gen=max_gen_num,
                                                               min_num_features=min_num_features,
                                                               sampling=sampling_strategy,
                                                               total_num_features=total_num_features,
                                                               x_train=x_train, y_train=y_train,
                                                               x_valid=x_valid, y_valid=y_valid,
                                                               verbose=False)

    # save recall_list, the best solution, best recall as csv file
    recall_df = pd.DataFrame({"recall": recall_list})
    recall_df.to_csv("./Experiment_result/Recall Error ComparisonGroup " + str(sampling_strategy) + ".csv", index=False)

    best_solution = pd.DataFrame([best_solution])
    best_solution.to_csv("./Experiment_result/BestSolution " + str(sampling_strategy) + ".csv", index=False, header=False)

    logging.info("Evolutions completed, the result has been saved!")

