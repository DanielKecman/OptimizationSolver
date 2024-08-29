from enum import Enum
import cvxpy as cp

class ObjectiveFunctionType(Enum):
    MIN = "MIN",
    MAX = "MAX"


class OptimizationSolver:
    def __init__(self,
                 objective_function_type,
                 coeff_matrix,
                 coeff_vector,
                 decision_vars,
                 constraints):
        self._objective_function_type = objective_function_type
        self._coeff_matrix = coeff_matrix
        self._coeff_vector = coeff_vector
        self._decision_vars = decision_vars
        self._constraints = constraints

        self._validate_problem()

    def solve_problem (self):
        obj_func = None
        if self._objective_function_type == ObjectiveFunctionType.MIN:
            obj_func = cp.Minimize(0.5 * cp.quad_form(self._decision_vars, self._coeff_matrix)
                                    + self._coeff_vector.T @ self._decision_vars)
        else:
            obj_func = cp.Maximize(0.5 * cp.quad_form(self._decision_vars, self._coeff_matrix)
                                    + self._coeff_vector.T @ self._decision_vars)
        problem = cp.Problem(obj_func, self._constraints)
        return problem.solve()



    def _validate_problem(self):
        (rows, cols) = self._coeff_matrix.shape

        if rows != cols:
            raise ValueError(f"Expected square coefficient matrix but given matrix with dimensions ({rows}, {cols})")

        if rows != self._decision_vars.size:
            raise ValueError(f"Expected same dimension for coefficient matrix and number of decision variables, "
                             f"but given {rows} rows in matrix and {self._decision_vars.size} decision variables")

        if rows != self._coeff_vector.size:
            raise ValueError(f"Expected same dimension for coefficient vector and number of decision variables, "
                             f"but given {self._coeff_vector.size} vector length and {self._decision_vars.size} decision variables")

        for constraint in self._constraints:

            constraint_args = constraint.args
            lhs_expr = constraint_args[0]
            rhs_expr = constraint_args[1]

            A = lhs_expr.args[0].value
            b = rhs_expr.value

            (n_rows, n_cols) = A.shape

            if n_cols != self._decision_vars.size:
                raise ValueError(f"Expected equal number of columns in constraint matrix and number of variables"
                                 f"but given {n_cols} columns and {self._decision_vars.size} variables")

            if n_rows != b.size:
                raise ValueError(f"Expected equal number of rows in constraint matrix and length of constraint"
                                 f"vector but given {n_rows} rows and {b.size} constants")