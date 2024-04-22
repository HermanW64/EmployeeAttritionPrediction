from NSGA.ProblemDefine import MyProblem
from NSGA.TerminationCriteria import termination_criteria
from NSGA.Hyperparameters import set_nsga
from NSGA.Evolution import evolve
import logging

logging.basicConfig(level="INFO")


def repeat_evolution(sampling, total_repeat=1, max_gen=100, min_num_features=1,
                     total_num_features=None, x_train=None, y_train=None,
                     x_valid=None, y_valid=None, verbose=True):
    """
    :param sampling:
    :param total_num_features:
    :param total_repeat:
    :param max_gen:
    :param min_num_features:
    :param y_train:
    :param verbose:
    :param x_valid:
    :param y_valid:
    :param x_train:
    :return: the best solution (binary list)
    """

    # prepare container to record result from each evolution
    f1_score_error_list = []
    solution_list = []
    hv_list = []

    # 1. problem define:
    problem = MyProblem(total_number_features=total_num_features,
                        min_num_features=min_num_features,
                        x_train=x_train, y_train=y_train,
                        x_valid=x_valid, y_valid=y_valid)

    # 2. termination criteria
    termination = termination_criteria(max_gen=max_gen)

    # 3. set hyperparameters for NSGA (all parameters are already set)
    algorithm = set_nsga()

    # 4. run and visualize result of NSGA
    cycle = 1
    while cycle <= total_repeat:
        logging.info("\n----the cycle {0} begins----".format(cycle))
        try:
            min_f1_error, solution_binary, hv_value = evolve(problem=problem,
                                                             termination=termination,
                                                             algorithm=algorithm,
                                                             sampling=sampling,
                                                             total_num_features=total_num_features,
                                                             cycle=cycle,
                                                             verbose=verbose)

            f1_score_error_list.append(min_f1_error)
            solution_list.append(solution_binary)
            hv_list.append(hv_value)

        except Exception as e:
            logging.info("Some errors occur: \n", str(e))

        finally:
            # go to the next cycle of evolution (another repeat)
            cycle += 1

    # after all repeats, get the lowest recall and the corresponding solution
    best_index = f1_score_error_list.index(min(f1_score_error_list))
    best_f1_error = f1_score_error_list[best_index]
    best_solution = solution_list[best_index]

    return f1_score_error_list, best_solution, best_f1_error
