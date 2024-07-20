"""User input/output module

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""


def input_num(user_str='Please enter number', limit=(float('-inf'), float('inf')), target_type=int,
              included_borders=False, round_to=6):
    """User numeric input"""
    incorrect_val = True
    while incorrect_val:
        try:
            user_num = target_type(input(user_str))
            if (limit[0] >= user_num >= limit[1] and not included_borders) or \
                    (limit[0] > user_num > limit[1] and included_borders):
                raise ValueError(f'Expected value from {limit[0]} to {limit[1]}')
        except ValueError as err:
            print('Entered value is incorrect! ' + str(err))
            incorrect_val = True
        except Exception as err:
            print('Unknown error! ' + str(err))
            incorrect_val = True
        else:
            user_num = user_num.__round__(round_to)
            incorrect_val = False
    return user_num
