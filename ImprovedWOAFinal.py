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


class ImprovedWhaleOptimization:
    """class implements the whale optimization algorithm as found at
    http://www.alimirjalili.com/WOA.html
    and
    https://doi.org/10.1016/j.advengsoft.2016.01.008
    """

    """initialize solutions uniform randomly in space"""

    def __init__(
        self, opt_func, constraints, nsols, b, a, a_step, ndim, maximize=False
    ):
        self._fe = 0
        self._ndim = ndim
        self._opt_func = opt_func  # fitness function -> solve method
        self._constraints = constraints  # lb, ub -> solve
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._nsols = nsols
        self._best_solutions = []
        self._sols = self._init_solutions(nsols)

    def get_solutions(self):
        """return solutions"""
        return self._sols

    def optimize(self, t, ngens, max_fe):
        """solutions randomly encircle, search or attack"""
        ranked_sol = self._rank_solutions(self._sols)
        best_sol = ranked_sol[0]
        # include best solution in next generation solutions
        new_sols = [best_sol]

        for s in ranked_sol[1:]:
            if np.random.uniform(0.0, 1.0) > 0.5:
                A = self._compute_A()
                norm_A = np.linalg.norm(A)
                if norm_A < 1.0:
                    new_s = self._encircle(s, best_sol, A)
                else:
                    ###select random sol
                    random_sol = self._sols[np.random.randint(self._sols.shape[0])]
                    new_s = self._search(s, random_sol, A)
            else:
                new_s = self._attack(s, best_sol)
            new_sols.append(self._constrain_solution(new_s))

        de_sol = []
        for s in new_sols:
            if self._fe >= max_fe:
                return

            # mutation
            indices = np.random.randint(0, self._sols.shape[0], (3))

            r1 = self._sols[indices[0]]
            r2 = self._sols[indices[1]]
            r3 = self._sols[indices[2]]

            f = np.random.uniform(0, 1)
            v = r1 + f * (r2 - r3)
            jrand = np.random.randint(self._sols.shape[1])

            u = v
            # crossover
            for j in range(len(s)):
                if j == jrand:
                    u[j] = v[j]
                else:
                    u[j] = s[j]

            # selection
            self._fe += 1
            fitness_s = self._opt_func(s)
            self._fe += 1
            fitness_u = self._opt_func(u)
            if self._maximize:
                if fitness_u > fitness_s:
                    de_sol.append(u)
                else:
                    de_sol.append(s)
            else:
                if fitness_u < fitness_s:
                    de_sol.append(u)
                else:
                    de_sol.append(s)

        self._sols = np.stack(de_sol)
        self._a -= self._a_step

    def EOBL(self, sol):
        ranked_sol = self._rank_solutions(sol)
        elite_sol = ranked_sol[0]
        opposition_sols = []
        lb = np.min(ranked_sol, axis=0)
        ub = np.max(ranked_sol, axis=0)

        for s in ranked_sol[0:]:
            k = np.random.uniform(0.0, 1.0, size=self._ndim)
            opp_sol = k * (lb + ub) - elite_sol
            constrain_opp_s = []
            for c, s in zip(self._constraints, opp_sol):
                if c[0] > s or c[1] < s:
                    s = np.random.uniform(lb, ub, size=self._ndim)
            constrain_opp_s.append(s)
            opposition_sols.append(constrain_opp_s)

        opposition_sols = np.stack(opposition_sols)

        return opposition_sols

    def _init_solutions(self, nsols):
        """initialize solutions uniform randomly in space"""
        sols = []
        sols = np.random.uniform(
            self._constraints[0], self._constraints[1], size=(nsols, self._ndim)
        )
        # print(sols)
        sols = np.stack(sols, axis=-1)
        opp_sols = self.EOBL(sols)

        sols = np.concatenate((sols, opp_sols[0]))
        fitness = []
        for s in sols:
            self._fe += 1
            fitness.append(self._opt_func(s))
        sol_fitness = [(f, s) for f, s in zip(fitness, sols)]
        ranked_sol = list(
            sorted(sol_fitness, key=lambda x: x[0], reverse=self._maximize)
        )
        eobl_sol = []

        for i in range(nsols):
            eobl_sol.append(ranked_sol[i][1])

        sols = np.stack(eobl_sol, axis=0)

        return sols

    def _constrain_solution(self, sol):
        """ensure solutions are valid wrt to constraints"""
        constrain_s = []

        # print(self._constraints)

        for i in range(self._ndim):
            if sol[i] < self._constraints[0][i]:
                sol[i] = self._constraints[0][i]
            if sol[i] > self._constraints[1][i]:
                sol[i] = self._constraints[1][i]

            constrain_s.append(sol[i])

        return constrain_s

        # for s in sol:
        #     print(s)
        #     if s < self._constraints[0]:
        #         s = self._constraints[0]
        #     if s > self._constraints[1]:
        #         s = self._constraints[1]

        # constrain_s.append(s)
        # return constrain_s

    def _rank_solutions(self, sol):
        """find best solution"""
        fitness = []
        for s in sol:
            self._fe += 1
            fitness.append(self._opt_func(s))
        sol_fitness = [(f, s) for f, s in zip(fitness, sol)]
        # best solution is at the front of the list
        ranked_sol = list(
            sorted(sol_fitness, key=lambda x: x[0], reverse=self._maximize)
        )
        self._best_solutions.append(ranked_sol[0])

        return [s[1] for s in ranked_sol]

    def best_solutions(self):
        # print('generation best solution history')
        # print('([fitness], [solution])')
        # for s in self._best_solutions:
        #     print(s)
        # print('\n')
        # print('best solution')
        # print('([fitness], [solution])')
        return sorted(self._best_solutions, key=lambda x: x[0], reverse=self._maximize)[
            0
        ]

    def _compute_A(self):
        r = np.random.uniform(0.0, 1.0, size=self._ndim)
        return (2.0 * np.multiply(self._a, r)) - self._a

    def _compute_C(self):
        return 2.0 * np.random.uniform(0.0, 1.0, size=self._ndim)

    def _encircle(self, sol, best_sol, A):
        D = self._encircle_D(sol, best_sol)
        return best_sol - np.multiply(A, D)

    def _encircle_D(self, sol, best_sol):
        C = self._compute_C()
        D = np.linalg.norm(np.multiply(C, best_sol) - sol)
        return D

    def _search(self, sol, rand_sol, A):
        D = self._search_D(sol, rand_sol)
        return rand_sol - np.multiply(A, D)

    def _search_D(self, sol, rand_sol):
        C = self._compute_C()
        return np.linalg.norm(np.multiply(C, rand_sol) - sol)

    def _attack(self, sol, best_sol):
        D = np.linalg.norm(best_sol - sol)
        L = np.random.uniform(-1.0, 1.0, size=self._ndim)
        return (
            np.multiply(np.multiply(D, np.exp(self._b * L)), np.cos(2.0 * np.pi * L))
            + best_sol
        )


class IWOA:
    def __init__(self, epoch, pop_size):
        self.epoch = epoch
        self.pop_size = pop_size
        self.name = "Improved WOA: "

    def solve(self, problem_dict, termination):
        opt_func = problem_dict["fit_func"]

        b = 0.5
        a = 2.0
        a_step = 0.06
        constraints = [problem_dict["lb"], problem_dict["ub"]]
        if problem_dict["minmax"] == "min":
            maximize = False
        else:
            maximize = True

        opt_alg = ImprovedWhaleOptimization(
            opt_func,
            constraints,
            self.pop_size,
            b,
            a,
            a_step,
            problem_dict["dim"],
            maximize,
        )

        for t in range(self.epoch):
            opt_alg.optimize(t, self.epoch, termination["max_fe"])
        return opt_alg.best_solutions()


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
                    "dim": dimension,
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
                        best_fitness, best_position = model.solve(
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
                        f"=========={model.name} => Function: {func_map_key_name}, Dimension: {dimension}, Population Size: {pop_size} :: Completed: {WOAResults[func_map_key_name][dimension][pop_size]}==========",
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
    final_result(IWOA)
    sys.stdout = original_stdout
