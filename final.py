import numpy as np
from mealpy.swarm_based.WOA import OriginalWOA
import opfunu

problem = opfunu.cec_based.F42014(ndim=10)


def fitness_func(x):
    return problem.evaluate(x)


problem_dict = {
    "fit_func": fitness_func,
    "lb": problem.lb,
    "ub": problem.ub,
    "minmax": "min",
    "log_to": "file",
    "log_file": "history.log",
}
pop_size = 10
model = OriginalWOA(epoch=13000, pop_size=pop_size)
term_dict = {"max_fe": 100000}
best_position, best_fitness = model.solve(problem_dict, termination=term_dict)
# for _ in range(1, 52):
#     run_result = np.append(run_result, best_fitness)
# WOAResults[func_map_key_name][dimension][pop_size] = np.mean(run_result)
