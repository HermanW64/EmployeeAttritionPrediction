import logging
import warnings
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from NSGA.Logit import fit_logit
from statsmodels.tools.sm_exceptions import ConvergenceWarning
logging.basicConfig(level="INFO")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings('always')


class MyProblem(ElementwiseProblem):
    """
    define the optimization problem, including:
    # of variables, # of objectives,
    lower and upper of variables [x1, x2]
    x1: total number of features selected 1 to feature_num
    x2: classification error with selected features 0 to 100
    """

    def __init__(self, total_number_features=None, min_num_features=5, x_train=None, y_train=None,
                 x_valid=None, y_valid=None):

        super().__init__(n_var=total_number_features,
                         n_obj=2,
                         n_ieq_constr=0,
                         # the xl and xu should be the same size of n_var
                         xl=np.zeros(total_number_features),
                         xu=np.ones(total_number_features))

        # X should include the constant column
        self.X_train = x_train.copy()
        self.Y_train = y_train.copy()
        self.X_valid = x_valid.copy()
        self.Y_valid = y_valid.copy()
        self.min_num_features = min_num_features

    def _evaluate(self, x, out, *args, **kwargs):
        # x is one-dimensional vector
        # Apply the binary threshold to x
        x_binary = np.where(x > 0.5, 1, 0)

        # minimize 1 - F1-Score (f1) and minimize the number of features (f2)
        # get the selected feature names
        f2 = np.sum(x_binary)
        if f2 < 1:
            random_index = np.random.randint(0, len(x_binary))
            x_binary[random_index] = 1

        selected_features = self.X_train.columns[x_binary == 1]

        # generate the dataset with selected features
        x_train_selected = self.X_train.loc[:, selected_features]
        x_valid_selected = self.X_valid.loc[:, selected_features]
        y_train = self.Y_train
        y_valid = self.Y_valid

        # calculate classification score on the X_train_selected
        # fit the logit model using statsmodels
        logit_result, score_dict, confusion_matrix = fit_logit(x_train=x_train_selected,
                                                               x_valid=x_valid_selected,
                                                               y_train=y_train,
                                                               y_valid=y_valid)
        p_values = logit_result.pvalues
        p_values_df = pd.DataFrame(p_values, columns=["p_value"])

        # f1 (the 1st objective function, not F1 score): 1 - F1-Score
        # f2 (the 2nd objective function): number of features
        f1 = 1 - score_dict["f1"]

        # set constraints for the problem
        # if (p_values_df['p_value'] >= 0.05).sum() / len(p_values) >= 0.1:
        #     # discourage model with a lot of insignificant variables
        #     penalty_f1 = 0.99
        #     penalty_f2 = 24
        #     out["F"] = [penalty_f1, penalty_f2]

        if np.isnan(p_values_df['p_value']).sum() >= 1:
            # discourage model with invalid p-value
            penalty_f1 = 0.99
            penalty_f2 = 24
            out["F"] = [penalty_f1, penalty_f2]

        elif sum(x_binary) <= self.min_num_features:
            # discourage model with too few variables
            penalty_f1 = 0.99
            penalty_f2 = 24
            out["F"] = [penalty_f1, penalty_f2]

        elif (score_dict["recall"] == 0) or (score_dict["precision"] == 0):
            penalty_f1 = 0.99
            penalty_f2 = 24
            out["F"] = [penalty_f1, penalty_f2]

        else:
            out["F"] = [f1, f2]

