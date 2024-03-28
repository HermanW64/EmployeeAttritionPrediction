from pymoo.termination import get_termination
import logging
logging.basicConfig(level="INFO")


def termination_criteria(max_gen=100):
    """
    set termination criteria for NSGA-II algorithm,
    usually it is the maximum generation
    """

    termination = get_termination("n_gen", max_gen)

    return termination
