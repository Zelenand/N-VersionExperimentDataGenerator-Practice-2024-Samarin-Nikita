"""Vote algorithm model to make a vote and save and load it

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""
import json
import os

from dataaccess import vote_algorithm_data_access
from nversionmodule.data_generator import NResult
from votemodule.module_importer import ModuleNotLoadedError, FunctionNotFoundInModuleError

__author__ = "Denis V. Gruzenkin"
__copyright__ = "Copyright 2021, Denis V. Gruzenkin"
__credits__ = ["Denis V. Gruzenkin"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Denis V. Gruzenkin"
__email__ = "gruzenkin.denis@good-look.su"
__status__ = "Production"


class VoteAlgorithm:
    """Vote algorithm model"""

    def __init__(self, name: str, vote_func_name: str, module_name: str, module_pkg: str = 'VoteAlgorithms',
                 new_id=None):
        """
        Vote algorithm constructor
        :param name: Name of module
        :param vote_func_name: Vote function name
        :param module_name: Module name
        :param module_pkg: Module package
        :param new_id: New vote algorithm id
        """
        self._vote_module = None
        self._module_pkg = __import__(f'{module_pkg}.{module_name}')

        if hasattr(self._module_pkg, module_name):
            try:
                self._vote_module = getattr(self._module_pkg, module_name)
            except ModuleNotFoundError as e:
                err_str = f'Module VoteAlgorithms.{module_name} cannot be loaded. Error: {e.msg}'
                print(err_str)
                raise ModuleNotFoundError(err_str)
            except Exception as e:
                print(e.__str__())
                raise e
            else:
                if self._vote_module is not None:
                    if hasattr(self._vote_module, vote_func_name):
                        self._vote_algorithm = getattr(self._vote_module, vote_func_name)
                        self._module_name = module_name
                        self._vote_func_name = vote_func_name
                        self._id = new_id
                        self.name = name
                    else:
                        err_str = f'There is no function {vote_func_name} in {module_name}'
                        print(err_str)
                        raise FunctionNotFoundInModuleError(err_str)
                else:
                    err_str = f'Cannot load module VoteAlgorithms.{module_name}'
                    print(err_str)
                    raise ModuleNotLoadedError(err_str)

                self._vote_result: list[dict] = []
        else:
            err_str = f'Python module was not found by path: VoteAlgorithms.{module_name}'
            print(err_str)
            raise ModuleNotFoundError(err_str)

    @property
    def vote_algorithm(self):
        return self._vote_algorithm

    @vote_algorithm.setter
    def vote_algorithm(self, vote_func):
        self._vote_algorithm = vote_func

    @property
    def vote_result(self):
        return self._vote_result

    @property
    def id(self):
        return self._id

    @property
    def module_name(self):
        return self._module_name

    @module_name.setter
    def module_name(self, new_path: str):
        if os.path.isfile(new_path):
            self._module_name = new_path
        else:
            raise ModuleNotFoundError(f'Cannot find module by path: {new_path}')

    def vote(self, nversions_results: list[list[NResult]]):
        """Use vote algorithm on nversions results"""
        self._vote_result = list()
        for iteration_result in nversions_results:
            print("Iteration:", nversions_results.index(iteration_result))
            for version_result in iteration_result:
                print(str(version_result))
            self._vote_result.append({'data': iteration_result, 'res': self._vote_algorithm(iteration_result)})

    def save_vote_algorithm(self):
        """Save vote algorithm to database"""
        result = vote_algorithm_data_access.save_vote_algorithm(self)
        if result is not None:
            self._id = result

    def save_vote_results(self):
        """Save vote results to database"""
        if self._id is None:
            self.save_vote_algorithm()

        if vote_algorithm_data_access.save_vote_results(self):
            if input('Do you want to load rows ids from DB? Y - Yes; any key - No: ').upper() == 'Y':
                for res in self._vote_result:
                    tmp_ids: list[int] = list()
                    for iteration_res in res['data']:
                        select_res = vote_algorithm_data_access.load_vote_result_id(self._id, iteration_res.id,
                                                                                    res['res'])
                        if len(select_res) > 0:
                            if len(select_res[0]) > 0:
                                tmp_ids.append(int(select_res[0][0]))
                    res['ids'] = tmp_ids

    def load_vote_results(self):
        # TODO: Дописать этот метод!
        pass

    @classmethod
    def load_algorithms(cls) -> list:
        """Load vote algorithms from database"""
        algorithms_list = list()
        q_set = vote_algorithm_data_access.load_algorithms()
        if len(q_set) > 0:
            common_err_msg = 'Row is skipped'
            for mdl in q_set:
                try:
                    m_id = int(mdl[0])
                except ValueError:
                    print(f'Incorrect algorithm identifier - {mdl[0]}. {common_err_msg}')
                    continue
                else:
                    m_alg_name = mdl[1]
                    m_func = mdl[2]
                    m_name = mdl[3]
                    m_code = mdl[4]
                    m_pkg = mdl[5]
                    # Если путь валидный, то присваиваем его переменной для инициализации нового алгоритма
                    if not os.path.isfile(f'{m_pkg}/{m_name}.py'):
                        # Иначе создаём файл с исходным кодом из БД по этому пути
                        with open(f'{m_pkg}/{m_name}.py', 'w') as mod_file:
                            mod_file.write(json.loads(m_code))
                    try:
                        algorithms_list.append(VoteAlgorithm(m_alg_name, m_func, m_name, m_pkg, m_id))
                    except ModuleNotFoundError as e:
                        print(f'Error: {e.msg}. {common_err_msg}')
                    except ModuleNotLoadedError as e:
                        print(f'Error: {e}. {common_err_msg}')
                    except FunctionNotFoundInModuleError as e:
                        print(f'Error: {e}. {common_err_msg}')
                    except Exception as e:
                        print(f'Error: {e}. {common_err_msg}')
        else:
            print('There are no algorithms modules to load')

        return algorithms_list

    def __str__(self):
        return f'{self.name} ({self._vote_func_name}) in {self._module_name}'
