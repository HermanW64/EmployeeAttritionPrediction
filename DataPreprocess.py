import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level="INFO")


def generate_corr_map(dataset=None, plot_name=None):
    """
    :param plot_name:
    :param dataset: should be in DataFrame format
    :return:
    """
    correlation_matrix = dataset.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.savefig("./Datasets/correlation_heatmap_" + str(plot_name) + ".png")


def split_data(transformed_dataset=None, random_seed=343):
    """
    split the dataset into 3 sets, and save them locally
    random_seed: a random integer
    """
    # split the dataset into train, test, and valid (6:2:2)
    y = transformed_dataset["Attrition"]
    x = transformed_dataset.drop("Attrition", axis=1)
    x = sm.add_constant(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=random_seed)

    # force to convert into float type and save them
    x_train = x_train.astype(float)
    x_valid = x_valid.astype(float)
    x_test = x_test.astype(float)
    y_train = y_train.astype(float)
    y_valid = y_valid.astype(float)
    y_test = y_test.astype(float)

    x_train.to_csv("./Datasets/x_train.csv")
    x_valid.to_csv("./Datasets/x_valid.csv")
    x_test.to_csv("./Datasets/x_test.csv")
    y_train.to_csv("./Datasets/y_train.csv")
    y_valid.to_csv("./Datasets/y_valid.csv")
    y_test.to_csv("./Datasets/y_test.csv")

    # calculate the number of features
    total_num_features = len(x.columns.tolist())

    return total_num_features


def process_data():
    """
    Data Preprocessor: load and transform dataset, making it more suitable to build a logit model
    Including: Onehot Encoding, Normalization
    Return: A cleaned and transformed dataset in csv format
    """
    # 1. read the dataset
    load_data = pd.read_csv("./Datasets/Employee_Attrition.csv")

    # select categorical and numeric variables
    selected_var_c = ["Attrition", "BusinessTravel", "Department", "JobRole", "MaritalStatus", "OverTime"]

    selected_var_n = ["EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "MonthlyIncome",
                      "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
                      "YearsWithCurrManager"]

    # 2. make categorical and numeric one
    relevant_data_c = load_data[selected_var_c]
    relevant_data_n = load_data[selected_var_n]

    # 3. One-hot encoding for categorical variables
    # remove some redundant variables, for example, if there are 3 values in a category, then only keep 2.
    relevant_data_c_encoded = pd.get_dummies(relevant_data_c, drop_first=True)

    # 4. normalize numeric variables
    relevant_data_n_n = round((relevant_data_n - relevant_data_n.mean())/relevant_data_n.std(), 4)

    # 5. concatenate two sub-datasets: categorical and numeric, and save the transformed dataset
    transformed_dataset = pd.concat([relevant_data_c_encoded, relevant_data_n_n], axis=1)

    # 6. remove some highly-correlated variables (based on correlation matrix)
    generate_corr_map(dataset=transformed_dataset, plot_name="before")

    rm_list = ["BusinessTravel_Travel_Rarely", "Department_Research & Development", "Department_Sales",
               "TotalWorkingYears", "YearsWithCurrManager", "YearsInCurrentRole"]
    transformed_dataset = transformed_dataset.drop(rm_list, axis=1)

    generate_corr_map(dataset=transformed_dataset, plot_name="after")

    # 7. save the final version
    transformed_dataset.to_csv("./Datasets/transformed_Employee_Attrition.csv", index=False)

    # 8. split the dataset into training set, validation set, and test set
    total_num_features = split_data(transformed_dataset=transformed_dataset, random_seed=343)
    logging.info("Dataset preprocessing completed! Dataset has been split!")

    return total_num_features



