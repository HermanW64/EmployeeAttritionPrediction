# This is the main file to run the whole program, from dataset split to model result analysis
# The program is built based on the flow chart in the paper
from DataPreprocess import process_data
from ControlGroup import generate_control_group
from OptimizeByStrategy import optimize_by_strategy
from ResultAnalysis import analyze_results
from FinalModel import build_final_model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # I. Dataset preprocess -> split datasets
    total_num_features = process_data()

    # II. no optimization with all features (as comparison), but with different sampling techniques
    generate_control_group()

    # III. Optimization
    # 1. set hyperparameters
    total_repeat_times = 31
    max_gen_num = 100
    min_num_features = 4

    # 2. run (can run for one particular sampling strategy at a time: none, under, over, smote)
    # Every time to run, just change the sampling strategy
    # See the result in folder Experiment_result
    optimize_by_strategy(sampling_strategy="none", total_num_features=total_num_features,
                         total_repeat_times=total_repeat_times, max_gen_num=max_gen_num,
                         min_num_features=min_num_features)

    optimize_by_strategy(sampling_strategy="under", total_num_features=total_num_features,
                         total_repeat_times=total_repeat_times, max_gen_num=max_gen_num,
                         min_num_features=min_num_features)

    optimize_by_strategy(sampling_strategy="over", total_num_features=total_num_features,
                         total_repeat_times=total_repeat_times, max_gen_num=max_gen_num,
                         min_num_features=min_num_features)

    optimize_by_strategy(sampling_strategy="smote", total_num_features=total_num_features,
                         total_repeat_times=total_repeat_times, max_gen_num=max_gen_num,
                         min_num_features=min_num_features)

    # IV. Gather the experiment result
    analyze_results()

    # V. build model with the best balancing technique and the best feature combination (smote)
    build_final_model(random_seed=343)



