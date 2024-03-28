from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import statsmodels.api as sm
import numpy as np


def fit_logit(x_train, x_valid, y_train, y_valid):
    """
    define the logit modeling function for the optimization problem
    feed training and test/validation set, calculate classification error, model details, and confusion matrix
    :parameter: training set and validation set
    :returns: model detail, score list, confusion matrix
    """
    # fit the logit model, calculate the accuracy
    logit_model = sm.Logit(y_train, x_train)
    logit_result = logit_model.fit(disp=False)

    # Make predictions on the validation set
    y_pred = logit_result.predict(x_valid.astype(float))
    y_pred_binary = (y_pred > 0.5).astype(int)

    # calculate confusion matrix
    cm = confusion_matrix(y_valid, y_pred_binary)
    recall = recall_score(y_valid, y_pred_binary, zero_division=np.nan)
    precision = precision_score(y_valid, y_pred_binary, zero_division=np.nan)
    f1 = f1_score(y_valid, y_pred_binary, zero_division=np.nan)
    accuracy = accuracy_score(y_valid, y_pred_binary)
    clf_error = 1 - accuracy
    score_dict = {"recall": recall, "precision": precision, "f1": f1, "accuracy": accuracy, "clf error": clf_error}

    return logit_result, score_dict, cm
