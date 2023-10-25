import os
import opfunu
import numpy as np
import pandas as pd
from mealpy.optimizer import Optimizer
from mealpy.swarm_based.WOA import OriginalWOA
import sys

original_stdout = sys.stdout

benchmark_dict = {
    2014: range(1, 31),
    2017: range(1, 30),
}


def final_result(model_name):
    WOAResults = {}
    for year in [2014, 2017]:
        for func_num in benchmark_dict[year]:
            func_map_key_name = f"F{func_num}{year}"
            WOAResults[func_map_key_name] = {}
            for dimension in [10, 30, 50, 100]:
                func_name = f"opfunu.cec_based.F42014(ndim={dimension})"
                WOAResults[func_map_key_name][dimension] = {}
                problem = eval(func_name)
                problem_dict = {
                    "fit_func": problem.evaluate,
                    "lb": problem.lb,
                    "ub": problem.ub,
                    "minmax": "min",
                    "log_to": "file",
                    "log_file": "history.log",
                }
                for pop_size in [
                    10,
                    20,
                    30,
                    40,
                    50,
                    70,
                    100,
                    200,
                    300,
                    400,
                    500,
                    700,
                    1000,
                ]:
                    run_result = np.array([])
                    for iter in range(1, 52):
                        epoch = 10000
                        pop_size = pop_size
                        model = model_name(epoch, pop_size)
                        max_fe = 10000 * dimension
                        term_dict = {"max_fe": max_fe}
                        best_position, best_fitness = model.solve(
                            problem_dict, termination=term_dict
                        )
                        run_result = np.append(run_result, best_fitness)
                        print(
                            f"{model.name} => Function: {func_map_key_name}, Dimension: {dimension}, Population Size: {pop_size} :: Iteration: {iter}, Fitness: {best_fitness}",
                            flush=True,
                        )
                    WOAResults[func_map_key_name][dimension][pop_size] = np.mean(
                        run_result
                    )
                    print(
                        f"=========={model.name} => Function: {func_map_key_name}, Dimension: {dimension}, Population Size: {pop_size} :: Completed==========",
                        flush=True,
                    )
    for key in WOAResults.keys():
        df = pd.DataFrame(WOAResults[key])
        directory_path = f"./results/{model.name}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        df.to_csv(f"{directory_path}/{key}.csv", index=True)


with open("./progress.txt", "w") as f:
    sys.stdout = f
    final_result(OriginalWOA)
    sys.stdout = original_stdout
