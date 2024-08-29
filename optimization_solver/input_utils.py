import numpy as np
from optimization_solver.optimization_solver import ObjectiveFunctionType
from optimization_solver.validation_utils import validate_positive_number


def get_int(prompt, validation_function = None):
    while True:
        val_str = input(prompt)
        try:
            val = int(val_str)
            if validation_function != None:
                validation_function(val)
            return val
        except Exception as e:
            print(f"You entered invalid value <{val_str}> ({repr(e)}), please re-enter valid value")


def get_float(prompt, validation_function = None):
    while True:
        val_str = input(prompt)
        try:
            val = float(val_str)
            if validation_function != None:
                validation_function(val)
            return val
        except Exception as e:
            print(f"You entered invalid value <{val_str}> ({repr(e)}), please re-enter valid value")


def get_matrix_input(ask, matrix_name):
    print(ask)
    number_of_rows = get_int(f"Enter the number of rows for {matrix_name}: ",
                             validate_positive_number)
    number_of_cols = get_int(f"Enter the number of columns for {matrix_name}: ",
                             validate_positive_number)

    print(f"Enter the elements for {matrix_name}:")
    matrix = []
    for i in range(number_of_rows):
        row = []
        for j in range(number_of_cols):
            row.append(get_float(f"{matrix_name}[{i},{j}] = "))
        matrix.append(row)
    return np.array(matrix)


def get_vector_input(ask, vector_name):
    print(ask)
    length = get_int(f"Enter the length of {vector_name}: ", validate_positive_number)

    print(f"Enter the elements for {vector_name}:")
    vector = []
    for i in range(length):
        vector.append(get_float(f"{vector_name}[{i}] = "))
    return np.array(vector)


def get_number_of_variables():
    num_vars = get_int("Enter the number of variables: ", validate_positive_number)
    return num_vars


def get_objective_function_type():
    while True:
        obj_type = input("Enter objective function Minimize or Maximize? (min/max): ").strip().lower()

        if obj_type == 'min':
            return ObjectiveFunctionType.MIN
        elif obj_type == 'max':
            return ObjectiveFunctionType.MAX
        print(f"Unrecognized objective function <{obj_type}>")


def get_constraints(decision_vars):
    constraints = []
    add_constraints = input("Do you want to add constraints? (yes/no): ").strip().lower()
    while add_constraints in ['y', 'yes']:
        constraint_matrix = get_matrix_input("Enter the constraint matrix A:", "A")
        constraint_vector = get_vector_input("Enter the constraint vector b:", "b")

        '''
        (n_rows, n_cols) = constraint_matrix.shape

        if n_cols != decision_vars.size:
            raise ValueError(f"Expected equal number of columns in constraint matrix and number of variables"
                             f"but given {n_cols} columns and {decision_vars.size} variables")

        if n_rows != constraint_vector.size:
            raise ValueError(f"Expected equal number of rows in constraint matrix and length of constraint"
                             f"vector but given {n_rows} rows and {constraint_vector.size} constants")
        '''
        constraint_type = input("Enter the type of constraint (<=, >=, ==): ").strip()
        if constraint_type == "<=":
            constraints.append(constraint_matrix @ decision_vars <= constraint_vector)
        elif constraint_type == ">=":
            constraints.append(constraint_matrix @ decision_vars >= constraint_vector)
        elif constraint_type == "==":
            constraints.append(constraint_matrix @ decision_vars == constraint_vector)
        else:
            print(f"Entry <{constraint_type}> is not valid constraint and will not be added")

        add_constraints = input("Do you want to add more constraints? (yes/no): ").strip().lower()
    return constraints
