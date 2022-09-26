"""Vote algorithm model to make a vote and save and load it

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""
import inspect
import json
import os
# from accessify import private

from data_base_connector import DBConnector
from data_generator import NResult
from module_importer import ModuleNotLoadedError, FunctionNotFoundInModuleError

__author__ = "Denis V. Gruzenkin"
__copyright__ = "Copyright 2021, Denis V. Gruzenkin"
__credits__ = ["Denis V. Gruzenkin"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Denis V. Gruzenkin"
__email__ = "gruzenkin.denis@good-look.su"
__status__ = "Production"


class VoteAlgorithm:
    _db_name = 'experiment.db'

    def __init__(self, name: str, vote_func_name: str, module_name: str, module_pkg: str = 'VoteAlgorithms',
                 new_id=None):
        self._vote_module = None
        self._module_pkg = __import__(f'{module_pkg}.{module_name}')

        # print(self._module_pkg.__dict__)
        if hasattr(self._module_pkg, module_name):
            try:
                self._vote_module = getattr(self._module_pkg, module_name)
                # self._vote_module = __import__(f'{module_pkg}.{module_name}')
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
    def vote_results(self):
        return self._vote_result

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
        self._vote_result = list()
        for iteration_result in nversions_results:
            self._vote_result.append({'data': iteration_result, 'res': self._vote_algorithm(iteration_result)})

    def save_vote_algorithm(self):
        cur_conn = DBConnector(self._db_name)
        if not cur_conn.table_exists('algorithm'):
            create_query = 'create table algorithm (id integer primary key autoincrement not null, '
            create_query += 'name varchar(127) not null, src_code text null, module_name varchar(255) not null, '
            create_query += 'func_name varchar(127) not null, module_pkg varchar(255) null);'
            cur_conn.execute_query(create_query, [], True, False)

        if self._id is None:
            insert_query = f"insert into algorithm (name, src_code, module_name, func_name, module_pkg) values ("
            insert_query += f"'{str(self.name)}', '{json.dumps(inspect.getsource(self._vote_module))}', "
            insert_query += f"'{self._module_name}', '{self._vote_func_name}', 'VoteAlgorithms');"
            # Возвращается список кортежей, берём первый элемент первого кортежа, т.к. id всего 1 возвращется!
            self._id = cur_conn.execute_query(insert_query, [], True)[0][0]
        else:
            update_query = f"update algorithm set name = '{self.name}', "
            update_query += f"src_code = '{json.dumps(inspect.getsource(self._vote_algorithm))}', "
            update_query += f"module_path = '{self._module_name}' where id = {self._id};"
            cur_conn.execute_query(update_query, [], True, False)

    def save_vote_results(self):
        cur_conn = DBConnector(self._db_name)
        if not cur_conn.table_exists('vote_result'):
            create_query = 'create table vote_result (id integer primary key autoincrement not null, '
            create_query += 'algorithm_id integer not null, experiment_data_id integer not null, '
            create_query += 'vote_answer real null, unique(algorithm_id, experiment_data_id, vote_answer) '
            create_query += 'on conflict replace, foreign key ("algorithm_id") references algorithm(id)'
            create_query += 'foreign key ("experiment_data_id") references experiment_data(id));'
            cur_conn.execute_query(create_query, [], True, False)

        if self._id is None:
            self.save_vote_algorithm()

        res_insert_lst = list()
        for res in self._vote_result:
            for iteration_res in res['data']:
                res_insert_lst.append((self._id, iteration_res.id, res['res']))
        if len(res_insert_lst) > 0:
            insert_query = 'insert into vote_result (algorithm_id, experiment_data_id, vote_answer) values(?,?,?);'
            cur_conn.execute_query(insert_query, res_insert_lst, True, False)

            if input('Do you want to load rows is from DB? y - Yes; any key - No: ').upper() == 'Y':
                for res in self._vote_result:
                    tmp_ids: list[int] = list()
                    for iteration_res in res['data']:
                        select_query = f"select id from vote_result where algorithm_id = {self._id} and "
                        select_query += f"experiment_data_id = {iteration_res.id} and vote_answer = {res['res']};"

                        select_res = cur_conn.execute_query(select_query)
                        if len(select_res) > 0:
                            if len(select_res[0]) > 0:
                                tmp_ids.append(int(select_res[0][0]))
                    res['ids'] = tmp_ids

    def load_vote_results(self):
        # TODO: Дописать этот метод!
        pass

    @classmethod
    def load_algorithms(cls) -> list:
        cur_conn = DBConnector(cls._db_name)
        algorithms_list = list()
        if cur_conn.table_exists('algorithm'):
            select_query = 'select id, name, func_name, module_name, src_code, module_pkg from algorithm order by id;'
            q_set = cur_conn.execute_query(select_query)
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
        else:
            raise LookupError(
                f'There is no "algorithm" table in {cls._db_name} data base. Save some algorithm before load it.')

        return algorithms_list

    def __str__(self):
        return f'{self.name} ({self._vote_func_name}) in {self._module_name}'
