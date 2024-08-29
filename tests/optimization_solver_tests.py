import unittest
import numpy as np
import cvxpy as cp
from optimization_solver.optimization_solver import OptimizationSolver, ObjectiveFunctionType

class OptimizationSolverTest(unittest.TestCase):
    def test_bad_run_one(self) -> None:
        objective_function_type = ObjectiveFunctionType.MAX
        coeff_matrix = np.array([[1, 0], [0, 1]])
        coeff_vector = np.zeros(2)
        decision_vars = cp.Variable(2)
        constraints = []
        solver = OptimizationSolver(objective_function_type,
                                    coeff_matrix,
                                    coeff_vector,
                                    decision_vars,
                                    constraints)
        self.assertRaises(Exception, solver.solve_problem)

    def test_good_run_one(self) -> None:
        objective_function_type = ObjectiveFunctionType.MIN
        coeff_matrix = np.array([[1, 0], [0, 1]])
        coeff_vector = np.zeros(2)
        decision_vars = cp.Variable(2)
        constraints = []
        solver = OptimizationSolver(objective_function_type,
                                    coeff_matrix,
                                    coeff_vector,
                                    decision_vars,
                                    constraints)
        (result, status) = solver.solve_problem()
        self.assertEqual(0, result)

    def test_good_run_quad_prob_lin_constraints(self) -> None:
        objective_function_type = ObjectiveFunctionType.MIN
        coeff_matrix = np.array([[2, 0], [0, 2]])
        coeff_vector = np.array([-1, -1])
        decision_vars = cp.Variable(2)
        A = np.array([[1,2], [-1,-2], [-1,0], [0,-1]])
        b = np.array([1,-1,0,0])
        constraints = [A @ decision_vars <= b]
        solver = OptimizationSolver(objective_function_type,
                                    coeff_matrix,
                                    coeff_vector,
                                    decision_vars,
                                    constraints)
        (result, status) = solver.solve_problem()
        self.assertEqual(-0.45, result)

    def test_good_run_linear_prob(self) -> None:
        objective_function_type = ObjectiveFunctionType.MIN
        coeff_matrix = np.zeros(shape=[2,2])
        coeff_vector = np.array([1,2])
        decision_vars = cp.Variable(2)
        A = np.array([[-1,1], [1,2], [2,1]])
        b = np.array([1,4,5])
        constraints = [A @ decision_vars <= b]
        solver = OptimizationSolver(objective_function_type,
                                    coeff_matrix,
                                    coeff_vector,
                                    decision_vars,
                                    constraints)
        (result, status) = solver.solve_problem()
        self.assertEqual('unbounded', status)

    def test_good_run_quad_prob_eq_constraints(self) -> None:
        objective_function_type = ObjectiveFunctionType.MIN
        coeff_matrix = np.array([[4, 1], [1, 2]])
        coeff_vector = np.array([1,1])
        decision_vars = cp.Variable(2)
        A = np.array([[1,1]])
        b = np.array([1])
        constraints = [A @ decision_vars == b]
        solver = OptimizationSolver(objective_function_type,
                                    coeff_matrix,
                                    coeff_vector,
                                    decision_vars,
                                    constraints)
        (result, status) = solver.solve_problem()
        self.assertEqual(1.875, result)

    def test_good_run_max_linear_prob(self) -> None:
        objective_function_type = ObjectiveFunctionType.MAX
        coeff_matrix = np.zeros(shape=[2,2])
        coeff_vector = np.array([3,2])
        decision_vars = cp.Variable(2)
        A = np.array([[1,2], [3,1]])
        b = np.array([4,6])
        constraints = [A @ decision_vars <= b]
        solver = OptimizationSolver(objective_function_type,
                                    coeff_matrix,
                                    coeff_vector,
                                    decision_vars,
                                    constraints)
        (result, status) = solver.solve_problem()
        self.assertEqual(7.2, result)

if __name__ == '__main__':
    unittest.main()
