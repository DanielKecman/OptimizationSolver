import cvxpy as cp
from optimization_solver.input_utils import (get_matrix_input,
                                             get_vector_input,
                                             get_constraints,
                                             get_objective_function_type,
                                             get_number_of_variables)
from optimization_solver.optimization_solver import OptimizationSolver


if __name__ == "__main__":

    print("This solver solves optimization problems of the form:")
    print("min(x) 1/2 x'Qx + c'x")
    print("such that Ax <= b or Ax == b or Ax >= b")
    print("with the option to set objective function for maximization")

    try:
        objective_function_type = get_objective_function_type()

        num_vars = get_number_of_variables()
        decision_vars = cp.Variable(num_vars)

        matrix_input = get_matrix_input("Enter the quadratic coefficient matrix Q:", "Q")
        vector_input = get_vector_input("Enter the linear coefficient vector c:", "c")

        constraints = get_constraints(decision_vars)

        solver = OptimizationSolver(objective_function_type,
                                    matrix_input,
                                    vector_input,
                                    decision_vars,
                                    constraints)
        (result, status) = solver.solve_problem()

        print()
        print("Optimization Complete!")
        print("Problem Status:", status)
        print("Optimal value:", result)
        print("Optimal variables:", decision_vars.value)

    except Exception as e:
        print(f"Optimization solver failed due to : {repr(e)}")


    