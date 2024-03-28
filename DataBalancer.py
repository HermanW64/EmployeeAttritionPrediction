import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


def balance_data(sampling="none", random_seed=343):
    """
    balance the training set data using particular strategy
    :param random_seed: default 343
    :param sampling: {"none", "under", "over", "smote"}
    :return:
    """
    # read the training datasets
    x_train = pd.read_csv("./Datasets/x_train.csv", index_col=0)
    y_train = pd.read_csv("./Datasets/y_train.csv", index_col=0)

    # perform sampling techniques on training set only
    if sampling == "under":
        # separate attrition and not group
        training_set = pd.concat([x_train, y_train], axis=1)
        training_set_majority = training_set[training_set["Attrition"] == 0]
        training_set_minority = training_set[training_set["Attrition"] == 1]

        # Undersample the majority class to match the minority class
        undersampled_majority_dataset = training_set_majority.sample(n=len(training_set_minority))

        # Combine the under-sampled majority class with the minority class
        undersampled_dataset = pd.concat([undersampled_majority_dataset, training_set_minority], axis=0)
        x_train = undersampled_dataset.drop('Attrition', axis=1)
        y_train = undersampled_dataset['Attrition']

    elif sampling == "over":
        # Over-sample the minority class
        oversampler = RandomOverSampler(random_state=random_seed)
        x_train, y_train = oversampler.fit_resample(x_train, y_train)

    elif sampling == "smote":
        # perform SMOTE to generate balanced dataset
        smoter = SMOTE(random_state=random_seed)
        x_train, y_train = smoter.fit_resample(x_train, y_train)

    elif sampling == "none":
        pass

    else:
        print("Only smote, under, over are acceptable!")

    return x_train, y_train
