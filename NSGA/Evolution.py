from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
import logging
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
logging.basicConfig(level="INFO")


def plot_pareto(plot_name, sampling, total_num_features, gen_f):
    """

    :return:
    """
    title = str(plot_name) + "(" + str(sampling) + ")"
    plt.figure(figsize=(7, 5))
    plt.scatter(gen_f[:, 0], gen_f[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title(title)
    plt.xlabel("1 - recall")
    plt.xlim(0, 1)
    plt.ylabel("number of variables")
    plt.ylim(0, total_num_features)
    plt.savefig("./Pareto_Images/" + str(title) + ".png")


def evolve(problem, algorithm, termination, sampling, total_num_features=None, cycle=None, verbose=False):
    """
    Run optimization from generation 0 to max generation
    :param sampling:
    :param cycle:
    :param problem:
    :param algorithm:
    :param termination:
    :param total_num_features:
    :param verbose:
    :return:
    """
    # 1. run the predefined problem
    res = minimize(problem,
                   algorithm,
                   termination,
                   save_history=True,
                   verbose=verbose)

    # 2. save initial pareto and final pareto front
    # -- get the objectives f1 and f2 of the first and last generation
    initial_gen_f = res.history[0].pop.get("F")
    last_gen_f = res.history[-1].pop.get("F")

    # -- visualization (only plot for the first evolution)
    if cycle == 1:
        # show pareto front in the end
        plot_pareto(plot_name="Initial pareto", sampling=sampling, total_num_features=total_num_features, gen_f=initial_gen_f)

        # show pareto front in the end
        plot_pareto(plot_name="Final pareto", sampling=sampling, total_num_features=total_num_features, gen_f=last_gen_f)

    # 3. calculate HV
    # reference point should be the worst point among the population (1, total number of features)
    worst_obj1 = np.max(initial_gen_f[:, 0]).copy()
    worst_obj2 = np.max(initial_gen_f[:, 1]).copy()
    ref_point = np.array([worst_obj1, worst_obj2])

    # get all points of the best pareto front in the last generation
    ind = HV(ref_point=ref_point)
    hv_value = round(ind(initial_gen_f), 4)
    logging.info("HV value: " + str(hv_value))

    # 4. record the best solutions and minimum recall error
    # -- get the result data
    x = res.X
    f = np.round(res.F, 4)

    # -- find out the lowest recall_error from the result (validation set)
    min_recall_index = np.argmin(f[:, 0])
    min_recall = f[min_recall_index, 0]

    # -- find out the corresponding feature selection
    best_solution = x[min_recall_index, :]
    best_solution_binary = np.where(best_solution > 0.5, 1, 0)
    logging.info("minimum recall error on validation set: " + str(min_recall))
    logging.info("number of the corresponding selected features: " + str(sum(best_solution_binary)))

    return min_recall, best_solution_binary, hv_value
