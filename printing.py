def print_truncated(value, decimals=2):
    value_str = str(value)
    if '.' in value_str:
        integer_part, decimal_part = value_str.split('.')
        truncated_decimal = decimal_part[:decimals]
        print(f"{integer_part}.{truncated_decimal}")
    else:
        print(value_str + '.' + '0' * decimals)