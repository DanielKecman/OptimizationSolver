def validate_positive_number(value):
    if value <= 0:
        raise ValueError(f"Expected positive value, but got <{value}>")