import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import page_trend_test, wilcoxon
import logging
logging.basicConfig(level="INFO")


def analyze_results():
    """
    perform statistical tests
    :return:
    """
    # check if the analysis result is already existing

    # read experiment results into a pandas table
    recall_list_none = pd.read_csv("result backups 1227 2023/Recall Error ComparisonGroup none.csv")
    recall_list_under = pd.read_csv("result backups 1227 2023/Recall Error ComparisonGroup under.csv")
    recall_list_over = pd.read_csv("result backups 1227 2023/Recall Error ComparisonGroup over.csv")
    recall_list_smote = pd.read_csv("result backups 1227 2023/Recall Error ComparisonGroup smote.csv")
    recall_data = {"none": recall_list_none, "under": recall_list_under,
                   "over": recall_list_over, "smote": recall_list_smote}
    exp_df = pd.concat(recall_data.values(), keys=recall_data.keys(), ignore_index=True, axis=1)
    exp_df.columns = ["none", "under", "over", "smote"]
    exp_df.to_csv("./Experiment_result/Experiment result.csv", index=False)

    # read control group data
    file_path = "result backups 1227 2023/Recall Error ControlGroup.txt"  # Replace with the actual file path

    with open(file_path, 'r') as file:
        content = file.readlines()

    control_group_data = {}
    for line in content:
        key, value = line.strip().split(': ')
        control_group_data[key] = float(value)

    # sketch histogram to show the distribution of four categories
    plt.figure(figsize=(10, 10))
    num_bins = 20
    bin_edges = [i * (1.0 / num_bins) for i in range(num_bins + 1)]
    for column, color in zip(exp_df.columns, ['red', 'blue', 'green', 'orange']):
        plt.hist(exp_df[column], bins=bin_edges, alpha=0.25, color=color, label=column)

    line_colors = ['red', 'blue', 'green', 'orange']
    for (key, value), color in zip(control_group_data.items(), line_colors):
        plt.axvline(value, color=color, linestyle='dashed', linewidth=2, alpha=0.95, label=f'{key} and no optimized')

    plt.xlabel('1 - Recall')
    plt.ylabel('Frequency')
    plt.title('Distribution of 1 - Recall')
    plt.legend()
    plt.savefig('./Experiment_result/Recall Error distributions.png')

    logging.info("Experiment dataset saved and recall error (1 - recall) distributions generated!")

    # Q1: optimization effects on different sampling strategies (non-parametric ANOVA: JT test and Page test)
    desc = "Compare the median of recall error. It represents better performance when it is lower."
    h0 = "median(none) = median(under) = median(over) = median(smote)"
    h1 = "median(none) >= median(under) >= median(over) >= median(smote), at least one inequality is strict"
    res_page = page_trend_test(data=exp_df, predicted_ranks=[4, 3, 2, 1])
    l_stat = round(res_page.statistic, 4)
    p_value = round(res_page.pvalue, 4)

    if p_value < 0.05:
        conclusion = "L statistics is {0}, and P-value is {1}, therefore h0 is rejected!".format(str(l_stat), str(p_value))
    else:
        conclusion = "L statistics is {0}, and P-value is {1}, therefore h1 is rejected!".format(str(l_stat), str(p_value))

    page_test_path = "result backups 1227 2023/Test result Page Test.txt"

    # Save conclusions to the text file
    with open(page_test_path, 'w') as file:
        file.write(desc + "\n")
        file.write("H0: {}\n".format(h0))
        file.write("H1: {}\n".format(h1))
        file.write(conclusion)

    logging.info("Page test conclusion saved!")

    # Q2: optimization vs no optimization for each strategy (control sampling strategy)
    strategy_list = ["none", "under", "over", "smote"]
    wilcoxon_test_path = "result backups 1227 2023/Test result Wilcoxon Rank.txt"
    with open(wilcoxon_test_path, 'w') as file:
        for strategy in strategy_list:
            desc = "Compare the recall error of optimized models and that of not optimized one, for sampling strategy: {0}".format(strategy)
            h0 = "median of recall error from optimized models = median of recall from not optimized one"
            h1 = "median of recall error from optimized models < median of recall from not optimized one"
            res_wil = wilcoxon(x=exp_df[strategy], y=[control_group_data[strategy]]*31, alternative="less", method="approx")
            p_value = round(res_wil.pvalue, 4)
            if p_value < 0.05:
                conclusion = "P-value is {0}, therefore h0 is rejected!".format(str(p_value))
            else:
                conclusion = "P-value is {0}, therefore h1 is rejected!".format(str(p_value))

            file.write(desc + "\n")
            file.write("H0: {}\n".format(h0))
            file.write("H1: {}\n".format(h1))
            file.write(conclusion + "\n\n")

    logging.info("Wilcoxon rank test conclusion saved!")
