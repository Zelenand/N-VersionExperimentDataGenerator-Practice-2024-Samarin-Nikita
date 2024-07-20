"""Main execution file

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""

import sys

from nversionmodule.data_generator import NModule
from userinput.user_input import input_num
from votemodule.VoteAlgorithm import VoteAlgorithm


def check_is_none_var(var: any, err_str: str = 'Variable is None') -> bool:
    """None check"""
    result = var is not None
    if not result:
        print(err_str)
    return result


def show_list(lst: list[tuple], header='List items:'):
    """Print list"""
    print(header)
    for i in range(len(lst)):
        print(f'\n{i + 1}. {str(lst[i])}')
    print('\n')


def main():
    module_index_err_str = 'Current module index is not defined yet. Please set it before using this menu item.'
    menu_dict = {
        'Add module to list': 1,
        'Load module list from DB': 2,
        'Save module to DB': 3,
        'Show modules list': 4,
        'Set current module': 5,
        'Show current module': 6,
        'Add versions to module': 7,
        'Load module with versions from DB': 8,
        'Save module with versions to DB': 9,
        'Show module versions': 10,
        'Generate experiment data for module': 11,
        'Save module experiment data to DB': 12,
        'Load module experiment data from DB': 13,
        'Add vote algorithm': 14,
        'Load vote algorithms': 15,
        'Save vote algorithms list': 16,
        'Show vote algorithms': 17,
        'Run vote algorithms': 18,
        'Make experiment analysis': 19,
        'Exit': 0
    }
    menu_str = '\nPlease choice menu item:\n'
    for key, value in menu_dict.items():
        menu_str += f'{str(value)}. {key}\n'

    modules_list = []
    current_module_index: int = None

    vote_algorithms_list = []

    experiments_names_list = list()

    while (user_chosen_item := input_num(menu_str, (0, 12), int, True, 0)) != menu_dict['Exit']:
        if user_chosen_item == menu_dict['Add module to list']:
            add_module_to_list(modules_list)
        elif user_chosen_item == menu_dict['Load module list from DB']:
            tmp_module = NModule('NoName', 6)
            tmp_module.load_module()
            modules_list.append(tmp_module)
        elif user_chosen_item == menu_dict['Save module to DB']:
            current_module_index = save_module_to_db(current_module_index, modules_list)
        elif user_chosen_item == menu_dict['Show modules list']:
            show_list(modules_list)
        elif user_chosen_item == menu_dict['Set current module']:
            show_list(modules_list)
            current_module_index = input_num('Enter module order number to select it as a current module: ',
                                             (-1, len(modules_list))) - 1
            print(f'Current module index is {current_module_index + 1}')
        elif user_chosen_item == menu_dict['Show current module']:
            if check_is_none_var(current_module_index, module_index_err_str):
                print(f'{current_module_index}. {str(modules_list[current_module_index])}')
        elif user_chosen_item == menu_dict['Add versions to module']:
            if check_is_none_var(current_module_index, module_index_err_str):
                modules_list[current_module_index].add_versions()
        elif user_chosen_item == menu_dict['Load module with versions from DB']:
            load_module_with_versions_from_db(current_module_index, modules_list)
        elif user_chosen_item == menu_dict['Save module with versions to DB']:
            if check_is_none_var(current_module_index, module_index_err_str):
                modules_list[current_module_index].save_module_with_versions()
        elif user_chosen_item == menu_dict['Show module versions']:
            if check_is_none_var(current_module_index, module_index_err_str):
                show_list(modules_list[current_module_index].versions_list,
                          f'Versions of {current_module_index + 1} module')
        elif user_chosen_item == menu_dict['Generate experiment data for module']:
            generate_experiment_data_for_module(current_module_index, experiments_names_list, module_index_err_str,
                                                modules_list)
        elif user_chosen_item == menu_dict['Save module experiment data to DB']:
            if check_is_none_var(current_module_index, module_index_err_str):
                modules_list[current_module_index].save_experiment_data()
        elif user_chosen_item == menu_dict['Load module experiment data from DB']:
            experiments_names_list = load_module_experiment_data_from_db(current_module_index, experiments_names_list,
                                                                         module_index_err_str, modules_list)
        elif user_chosen_item == menu_dict['Add vote algorithm']:
            algorithm_name = input('Enter vote algorithm name: ')
            module_name = input('Enter python module name (it should located in VoteAlgorithms package): ')
            vote_function_name = input('Enter vote function name in module: ')
            vote_algorithms_list.append(VoteAlgorithm(algorithm_name, vote_function_name, module_name))
        elif user_chosen_item == menu_dict['Load vote algorithms']:
            usr_answer = input('Unsaved algorithms will be lost! Are you sure (Y - yes; any key - no)? ')
            if usr_answer.upper() == 'Y':
                vote_algorithms_list = VoteAlgorithm.load_algorithms()
        elif user_chosen_item == menu_dict['Save vote algorithms list']:
            if len(vote_algorithms_list) > 0:
                for cur_algorithm in vote_algorithms_list:
                    cur_algorithm.save_vote_algorithm()
            else:
                print('There are no algorithms to save!')
        elif user_chosen_item == menu_dict['Show vote algorithms']:
            if len(vote_algorithms_list) > 0:
                for cur_algorithm in vote_algorithms_list:
                    print(f'{cur_algorithm}')
            else:
                print('There are no algorithms to show!')
        elif user_chosen_item == menu_dict['Run vote algorithms']:
            run_vote_algorithms(current_module_index, module_index_err_str, modules_list, vote_algorithms_list)
        elif user_chosen_item == menu_dict['Make experiment analysis']:
            pass


def run_vote_algorithms(current_module_index, module_index_err_str, modules_list, vote_algorithms_list):
    if len(modules_list[current_module_index].global_results_lst) > 0:
        if check_is_none_var(current_module_index, module_index_err_str):
            if len(vote_algorithms_list) > 0:
                for cur_algorithm in vote_algorithms_list:
                    print(cur_algorithm.vote_algorithm.__name__)
                    cur_algorithm.vote(modules_list[current_module_index].global_results_lst)
                    cur_algorithm.save_vote_results()
            else:
                print('There are no algorithms to show!')
        else:
            print('Current module does not save id DB! Save all necessary data before running algorithms')
    else:
        print('There is no experiment data list. Please generate it before run voting!')


def load_module_experiment_data_from_db(current_module_index, experiments_names_list, module_index_err_str,
                                        modules_list):
    if check_is_none_var(current_module_index, module_index_err_str):
        if len(experiments_names_list) == 0:
            print('I am loading experiments names from DB. It can take several minutes...')
            experiments_names_list = modules_list[current_module_index].get_experiments_names()
        show_list(experiments_names_list, 'Experiments names:')
        exp_name_index = input_num('Choice experiment by order number to its data load: ',
                                   (-1, len(modules_list))) - 1
        try:
            modules_list[current_module_index].load_experiment_data(experiments_names_list[exp_name_index])
        except ValueError:
            print('There is no module name to load!')
        except Exception:
            print('Something is wrong!')
    return experiments_names_list


def generate_experiment_data_for_module(current_module_index, experiments_names_list, module_index_err_str,
                                        modules_list):
    if check_is_none_var(current_module_index, module_index_err_str):
        exp_name = input('Enter experiment name: ')
        generate_data = modules_list[current_module_index].generate_experiment_data(
            input_num('Enter iterations amount: ', (0.0, float('inf'))), exp_name)

        if generate_data is not None:
            print('Experiment data was generated successfully!')
            experiments_names_list.append(exp_name)
        else:
            print('Error occurs during experiment data generation.')


def load_module_with_versions_from_db(current_module_index, modules_list):
    if current_module_index is not None:
        modules_list[current_module_index].load_module_with_versions()
    else:
        mdl = NModule('NoName', 6)
        mdl.load_module_with_versions()
        modules_list.append(mdl)


def save_module_to_db(current_module_index, modules_list):
    if len(modules_list) > 0:
        if current_module_index is None:
            show_list(modules_list)
            saving_index = input_num('Enter module order number to save it: ', (-1, len(modules_list)))
            current_module_index = saving_index - 1
        modules_list[current_module_index].save_module()
    else:
        print('Modules list is empty. There is no data to save!')
    return current_module_index


def add_module_to_list(modules_list):
    default_module_name = f'Module {len(modules_list) + 1}'
    new_module_name = input(f'Enter module name [{default_module_name}]: ')
    if new_module_name == '' or new_module_name is None:
        new_module_name = default_module_name
    new_module_round_digits = input_num('Enter number of module digits round: ', (0, float('inf')), int, True)
    new_model_min_generated_val = input_num('Enter minimal value that can be generated by versions: ',
                                            (0, float('inf')), float, False, new_module_round_digits)
    new_model_max_generated_val = input_num('Enter maximal value that can be generated by versions: ',
                                            (0, float('inf')), float, False, new_module_round_digits)
    modules_list.append(NModule(new_module_name,
                                new_module_round_digits,
                                new_model_min_generated_val,
                                new_model_max_generated_val))
    print(f'Module "{new_module_name}" was successfully added to the list!')


if __name__ == '__main__':
    sys.exit(main())
