"""Experiment data generation module to simulate N-version programming module running

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""
import json
from math import sqrt
from random import random, uniform, normalvariate

from data_base_connector import DBConnector

__author__ = "Denis V. Gruzenkin"
__copyright__ = "Copyright 2021, Denis V. Gruzenkin"
__credits__ = ["Denis V. Gruzenkin"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Denis V. Gruzenkin"
__email__ = "gruzenkin.denis@good-look.su"
__status__ = "Production"


def input_num(user_str='Please enter number', limit=(float('-inf'), float('inf')), target_type=int,
              included_borders=False, round_to=6):
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


class NVersion:
    """
    Real NVP version emulation
    """
    db_name = 'experiment.db'
    formal_json_db_lst_name = 'lst'

    def __init__(self, n_id: int = None, name: str = 'NoName', const_diversities: tuple[float] = tuple()
                 , dynamic_diversities: list[tuple] = None, reliability: float = 0):
        """
        NVersion class constructor
        :param n_id:
        :param name:
        :param const_diversities:
        :param dynamic_diversities:
        :param reliability:
        :type dynamic_diversities: list[tuple]
        """
        if dynamic_diversities is None:
            dynamic_diversities = list()
        self._id = n_id
        self._name = name
        self._const_diversities = const_diversities
        self._dynamic_diversities = dynamic_diversities
        self._reliability = reliability

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name.__str__()

    @property
    def const_diversities(self):
        return self._const_diversities

    @const_diversities.setter
    def const_diversities(self, new_tpl: tuple):
        self._const_diversities = new_tpl

    @property
    def dynamic_diversities(self):
        return self._dynamic_diversities

    @dynamic_diversities.setter
    def dynamic_diversities(self, new_lst: list):
        self._dynamic_diversities = new_lst

    @property
    def common_coordinates_list(self):
        return tuple(list(self._const_diversities) + self._dynamic_diversities)

    @property
    def reliability(self):
        return self._reliability

    def generate_reliability(self, min_val: float, max_val: float, round_to=6):
        self._reliability = uniform(min_val, max_val).__round__(round_to)

    @staticmethod
    def _calc_euclidean_distance(lst1: list[float], lst2: list[float]) -> float:
        if len(lst1) == len(lst2):
            square_dif_sum = 0
            for i in range(len(lst1)):
                square_dif_sum += (lst1[i] - lst2[i]) ** 2
            return sqrt(square_dif_sum)
        else:
            raise ValueError('Different coordinates amount')

    @reliability.setter
    def reliability(self, new_val: float):
        if 0 <= float(new_val) <= 1:
            self._reliability = new_val
        else:
            raise ValueError('Reliability interval is [0, 1]')

    @property
    def distance_from_zero_point(self):
        return self._calc_euclidean_distance(self.common_coordinates_list,
                                             list(map(lambda arg: arg * 0, self.common_coordinates_list)))

    # def generate_dynamic_diversities(self, count: int, min_val: float, max_val: float, round_to=6):
    #     self._dynamic_diversities = [uniform(min_val, max_val).__round__(round_to) for _ in range(count)]
    def generate_dynamic_diversities(self, intervals_lst: list[tuple], round_to=6):
        for tpl in intervals_lst:
            self._dynamic_diversities.append(uniform(tpl[0], tpl[1]).__round__(round_to))

    def calculate_distance_to(self, another_version) -> float:
        return self._calc_euclidean_distance(self.common_coordinates_list, another_version.common_coordinates_list)

    def __str__(self):
        return f'{self._id}. {self._name} [{self._reliability}] {self.common_coordinates_list}'

    def save(self, module_id=None):
        cur_conn = DBConnector(self.db_name)
        if not cur_conn.table_exists('version'):
            create_query = 'create table version ("id" integer primary key autoincrement not null, '
            create_query += '"name" varchar(255) not null, "const_diversities_coordinates" varchar(512) null, '
            create_query += '"dynamic_diversities_coordinates" varchar(512) null, "reliability" real not null, '
            create_query += '"module" integer null, foreign key ("module") references module(id));'
            cur_conn.execute_query(create_query, [], True, False)

        if self._id is None:
            additional_query_str = ', NULL'
            if module_id is not None:
                additional_query_str = f', {module_id}'

            insert_query = f"insert into version (name, const_diversities_coordinates, dynamic_diversities_coordinates,"
            insert_query += f" reliability, module) values ('{self._name}', "
            insert_query += f"'{json.dumps({self.formal_json_db_lst_name: self._const_diversities})}', "
            insert_query += f"'{json.dumps({self.formal_json_db_lst_name: self._dynamic_diversities})}', "
            insert_query += f"{self._reliability}" + additional_query_str + ");"
            print(insert_query)
            # Т.к. у нас возвращается список кортежей, берём первый элемент первого кортежа, т.к. id сего 1 возвращется!
            self._id = int(cur_conn.execute_query(insert_query, [], True)[0][0])
        else:
            update_query = f"update version set name = '{self.name}', "
            update_query += f"const_diversities_coordinates = "
            update_query += f"'{json.dumps({self.formal_json_db_lst_name: self._const_diversities})}', "
            update_query += f"dynamic_diversities_coordinates = "
            update_query += f"'{json.dumps({self.formal_json_db_lst_name: self._dynamic_diversities})}', "
            update_query += f"reliability = {self._reliability}, module = {module_id} where id = {self._id};"
            cur_conn.execute_query(update_query, [], True, False)

    def load(self):
        cur_conn = DBConnector('experiment.db')
        if cur_conn.table_exists('version'):
            if self._id is None:
                select_query = 'select name, const_diversities_coordinates, dynamic_diversities_coordinates, '
                select_query += 'reliability, module from version order by id;'
                q_set = cur_conn.execute_query(select_query)
                get_num_str = 'Choice version by id to load it.\n'
                for mdl in q_set:
                    get_num_str += f'\n{str(mdl)}'
                get_num_str += '\n'
                chosen_id = input_num(get_num_str)
            else:
                chosen_id = self._id

            select_query = f'select id, name, const_diversities_coordinates, dynamic_diversities_coordinates, '
            select_query += f'reliability, module from version where id = {chosen_id}'
            select_res = cur_conn.execute_query(select_query)

            # Т.к. у нас только 1 версия может быть найден по id, то обращаемся мы к 0-му индексу, без доп. проверок
            self._id = select_res[0][0]
            self.name = select_res[0][1]
            self._const_diversities = json.loads(select_res[0][2])[self.formal_json_db_lst_name]
            self._dynamic_diversities = json.loads(select_res[0][3])[self.formal_json_db_lst_name]
            self._reliability = float(select_res[0][4])
        else:
            raise LookupError(
                f'There is no "VERSION" table in {self.db_name} data base. Save module data before load it.'
            )

    @classmethod
    def load_versions_2_module(cls, module_id: int):
        if module_id is not None and type(module_id) == int:
            cur_conn = DBConnector(cls.db_name)
            if cur_conn.table_exists('version'):
                select_query = f'select id, name, const_diversities_coordinates, dynamic_diversities_coordinates, '
                select_query += f'reliability, module from version where module = {module_id}'
                select_res = cur_conn.execute_query(select_query)

                result_list = []
                print(select_res)
                for res in select_res:
                    tmp_ver = NVersion(res[0], res[1], json.loads(res[2])[cls.formal_json_db_lst_name],
                                       json.loads(res[3])[cls.formal_json_db_lst_name], float(res[4]))
                    result_list.append(tmp_ver)
                return result_list
            else:
                raise LookupError(
                    f'There is no "VERSION" table in {cls.db_name} data base. Save module data before load it.'
                )
        else:
            raise AttributeError(f'Invalid module_id parameter. Int is expected. {str(module_id)} was got.')


class NResult:
    def __init__(self, v_id: int, v_name: str, v_reliability: float, v_common_coordinates: tuple, v_answer: float
                 , c_answer: float, m_id: int, m_name: str, m_connectivity_matrix: tuple, m_iteration_num: int,
                 version: NVersion = None, res_id=None, experiment_name: str = 'NoName'):
        self.id = res_id
        self.version_id = v_id
        self.version_name = v_name
        self.version_reliability = v_reliability
        self.version_common_coordinates = v_common_coordinates
        self.version_answer = v_answer
        self.correct_answer = c_answer
        self.module_id = m_id
        self.module_name = m_name
        self.module_connectivity_matrix = m_connectivity_matrix
        self.module_iteration_num = m_iteration_num
        self.version = version
        self.experiment_name = experiment_name

    def __str__(self):
        result_str = f'{self.module_iteration_num}. {self.module_name} - {self.version_name} '
        result_str += f'({self.version_reliability}): Correct: {self.correct_answer}; Version answer: '
        result_str += f'{self.version_answer}'
        return result_str


class NModule:
    """
    N-version programming module
    """

    def __init__(self, name, round_to, min_out_val=100, max_out_val=1000):
        """
        NModule class constructor
        :param name: Module name
        :param round_to: Digits amount fallow after dot delimiter
        """
        self._id = None
        self.name = name
        self.round_to = round_to
        self.min_out_val = min_out_val
        self.max_out_val = max_out_val
        # Versions, which are included into module
        self._versions_list: list[NVersion] = []
        # List of tuples, that contain constant diversities values
        self._const_diversities_versions_list = []
        # List of tuples, that contain intervals to generate dynamic diversities
        self._dynamic_diversities_intervals_dict = dict()
        self._const_diversities_count = 0
        self._dynamic_diversities_count = 0
        self._global_results_lst: list = []
        self._global_results_lst_2_write: list[tuple] = []
        self._experiment_name = None

        self._db_name = 'experiment.db'

    @property
    def id(self):
        return self._id

    @property
    def versions_list(self):
        return self._versions_list

    @versions_list.setter
    def versions_list(self, new_ver_lst: list):
        self._versions_list = new_ver_lst

    @property
    def const_diversities_versions_list(self):
        return self._const_diversities_versions_list

    @const_diversities_versions_list.setter
    def const_diversities_versions_list(self, new_div_ver_lst: list):
        self._const_diversities_versions_list = new_div_ver_lst

    @property
    def dynamic_diversities_intervals_dict(self):
        return self._dynamic_diversities_intervals_dict

    @dynamic_diversities_intervals_dict.setter
    def dynamic_diversities_intervals_dict(self, new_dyn_div_intervals: list):
        self._dynamic_diversities_intervals_dict = new_dyn_div_intervals

    @property
    def const_diversities_count(self):
        return self._const_diversities_count

    @const_diversities_count.setter
    def const_diversities_count(self, const_div_count: int):
        self._const_diversities_count = const_div_count

    @property
    def dynamic_diversities_count(self):
        return self._dynamic_diversities_count

    @dynamic_diversities_count.setter
    def dynamic_diversities_count(self, dyn_div_count: int):
        self._dynamic_diversities_count = dyn_div_count

    @property
    def normed_connectivity_matrix(self):
        """
        Figure out distance length between versions in the metric space in percents (from 0 to 1)
        :return: matrix (tuple of tuples)
        """

        if len(self._versions_list) > 0:
            matrix = []
            for cur_ver in self._versions_list:
                matrix.append([cur_ver.calculate_distance_to(another_ver) for another_ver in self._versions_list])

            max_val = matrix[0][0]
            for row in matrix:
                if max(row) > max_val:
                    max_val = max(row)

            try:
                for i in range(len(matrix)):
                    for j in range(len(matrix[i])):
                        matrix[i][j] /= max_val
            except ZeroDivisionError:
                print('Max distance between all versions is zero, so connectivity matrix cannot by normed')

            for i in range(len(matrix)):
                matrix[i] = tuple(matrix[i])

            return tuple(matrix)
        else:
            raise AttributeError('There are not versions in module')

    @property
    def global_results_lst(self):
        return self._global_results_lst

    def _get_global_results_lst_2_write(self):
        for sub_lst in self._global_results_lst:
            for itm in sub_lst:
                self._global_results_lst_2_write.append(
                    (itm.version_id, itm.version_name, itm.version_reliability,
                     json.dumps({'version_coordinates': itm.version_common_coordinates}), itm.version_answer,
                     itm.correct_answer, itm.module_id, itm.module_name,
                     json.dumps({'connectivity_matrix': itm.module_connectivity_matrix}), itm.module_iteration_num,
                     itm.experiment_name)
                )

    @property
    def global_results_lst_2_write(self):
        if len(self._global_results_lst_2_write) == 0:
            self._get_global_results_lst_2_write()
        return self._global_results_lst_2_write

    def add_versions(self):
        while input('Add new version to module? Yes - any key; No - n. Your choice: ') != 'n':
            cur_new_version = NVersion()
            cur_new_version.name = input('Enter version name: ')

            if self._const_diversities_count < 1:
                self._const_diversities_count = input_num('Enter constant diversity metrics count: ', (0, float('inf')))
            if self._dynamic_diversities_count < 1:
                self._dynamic_diversities_count = input_num('Enter dynamic diversity metrics count: ',
                                                            (0, float('inf')))

            const_diversities_lst = []
            for i in range(self._const_diversities_count):
                const_diversities_lst.append(input_num(f'Enter {i + 1} diversity coordinate: ',
                                                       (float('-inf'), float('inf')),
                                                       float, self.round_to))
            cur_new_version.const_diversities = tuple(const_diversities_lst)
            self._const_diversities_versions_list.append(cur_new_version.const_diversities)

            tmp_dynamic_diversities_lst = []
            for i in range(self._dynamic_diversities_count):
                print(f'Enter limits (from min to max) to generate {i + 1} diversity coordinate')
                tmp_dynamic_diversities_lst.append(
                    (input_num('Min value: ', (float('-inf'), float('inf')), float, self.round_to),
                     input_num('Max value: ', (float('-inf'), float('inf')), float, self.round_to))
                )
            self._dynamic_diversities_intervals_dict[cur_new_version.name] = list(tmp_dynamic_diversities_lst)

            cur_new_version.reliability = input_num('Reliability: ', (0, 1), float, True, self.round_to)
            cur_new_version.generate_dynamic_diversities(self._dynamic_diversities_intervals_dict[cur_new_version.name],
                                                         self.round_to)

            self._versions_list.append(cur_new_version)

    def get_version_by_id(self, id_4_search: int) -> NVersion:
        for cur_ver in self._versions_list:
            if cur_ver.id is not None and cur_ver.id == id_4_search:
                return cur_ver
        else:
            return None

    def generate_experiment_data(self, iterations_amount: int, experiment_name: str) -> list:
        # Сначала находим версии-клоны - для них генерируем одно число для определения надёжности, если число >
        # минимальной надёжности, то версии выдают единый неверный результат, иначе - все выдают верный результат. Берём
        # версии, которые отличаются мало друг от друга и не входят в множество клонов, для них генерируем единое число
        # для определения надёжности, если среди них есть версии, выдавшие корректный результат, то берём этот результат
        # за базовый, иначе - генерируем базовый результат, вблизи которого ренерируем по нормальному распределению
        # ответы версий из этой группы. Находи группу версий, которы отличаются друг от друга примерно на 50%, для них
        # генерируем случаное число 0 или 1, если 0, то действем, как для предыдущей группы, иначе - как для следующей;
        # при этом генерируем число для диверсифицированности, чтобы определить,какие версии получат независимые ответы,
        # а какие сгенерированные по нормальному закону. Если версии различны > 50% примерно, то для каждой из них
        # генерируем разные вероятности на надёжностей и генерируем независимые ошибочные резульаты

        # Чтобы не возникало неопределённости при записи результатов в БД, очищаем имеющиеся результаты перед генерацией
        self._global_results_lst_2_write = list()
        self._global_results_lst = list()
        # Запускаем цикл по количеству зананных итераций
        for i in range(iterations_amount):
            result_lst = list()
            # Разбиваем версии на группы для различной генерации результатов их работы
            clone_versions = set()
            similar_versions = set()
            partly_similar_versions = set()
            difference_versions = set()
            for j in range(len(self._versions_list)):
                for k in range(len(self._versions_list)):
                    if k > j:  # Сравниваем версии только если они разыне и ещё не сравнивались
                        # Если версии отличаются не более чем на 5%, то считаем их клонами
                        if 0 <= self.normed_connectivity_matrix[j][k] <= 0.05:
                            if self._versions_list[k] not in difference_versions and \
                                    self._versions_list[k] not in similar_versions and \
                                    self._versions_list[k] not in partly_similar_versions:
                                clone_versions.add(self._versions_list[k])
                            if self._versions_list[j] not in difference_versions and \
                                    self._versions_list[j] not in similar_versions and \
                                    self._versions_list[j] not in partly_similar_versions:
                                clone_versions.add(self._versions_list[j])
                        elif 0.05 < self.normed_connectivity_matrix[j][k] < 0.4:  # Если версии похожи, но не клоны
                            if self._versions_list[j] not in difference_versions and \
                                    self._versions_list[j] not in clone_versions and \
                                    self._versions_list[j] not in partly_similar_versions:
                                similar_versions.add(self._versions_list[j])
                            if self._versions_list[k] not in difference_versions and \
                                    self._versions_list[k] not in clone_versions and \
                                    self._versions_list[k] not in partly_similar_versions:
                                similar_versions.add(self._versions_list[k])
                        elif 0.4 <= self.normed_connectivity_matrix[j][k] <= 0.6:  # Если версии частично похожи (~50%)
                            if self._versions_list[j] not in difference_versions and \
                                    self._versions_list[j] not in clone_versions and \
                                    self._versions_list[j] not in similar_versions:
                                partly_similar_versions.add(self._versions_list[j])
                            if self._versions_list[k] not in difference_versions and \
                                    self._versions_list[k] not in clone_versions and \
                                    self._versions_list[k] not in partly_similar_versions:
                                partly_similar_versions.add(self._versions_list[k])
                        else:  # Если версии имеют значительные различия
                            if self._versions_list[j] not in clone_versions and \
                                    self._versions_list[j] not in similar_versions and \
                                    self._versions_list[j] not in partly_similar_versions:
                                difference_versions.add(self._versions_list[j])
                            if self._versions_list[k] not in clone_versions and \
                                    self._versions_list[k] not in similar_versions and \
                                    self._versions_list[k] not in partly_similar_versions:
                                difference_versions.add(self._versions_list[k])

            # Генерируем значение, которое будет считаться правильным ответом на текущей итерации
            cur_correct_val = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)

            # Теперь для каждой группы версий генерируем результаты их работы
            # ---------------------------------------------------------------
            # 1. Генерируем выходные данные для версий-клонов
            min_rel = 1  # Устанавливаем максималльное значение надёжности для поиска минимума
            for ver in clone_versions:
                if ver.reliability < min_rel:
                    min_rel = ver.reliability
            # Генерируем число, которое покажет, отработали ли версии верно. Если оно будет больше минимальной
            # надёжности, то считаем, что все версии выдают одно ошибочное значение, иначе - все выдают верный ответ
            cur_clone_reliability = random()
            for ver in clone_versions:
                if cur_clone_reliability <= min_rel:
                    result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                              cur_correct_val, cur_correct_val, self._id, self.name,
                                              self.normed_connectivity_matrix, i, ver, None, experiment_name))
                else:
                    clone_error_val = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)
                    result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                              clone_error_val, cur_correct_val, self._id, self.name,
                                              self.normed_connectivity_matrix, i, ver, None, experiment_name))
            # 2. Генерируем выходные данные для явно схожих версий
            cur_similar_reliability = random()
            # Сначала в отдельном цикле ищем корректно отработавщие версии, чтобы в окрестностях верного ответа
            # генерировать неправильные результаты мультиверсий выдавщих ошибку. Если проводить проверку в одном цикле,
            # то когда первая из этого множества версий отработает некорректно, а некоторые другие - корректно, все
            # остальные версии будут генерировать ответы вблизи другого, случайно сгенерированного значения, что совсем
            # неверно, т.к. версии являются достаточно схожими
            correct_similar_versions = set()
            for ver in similar_versions:
                if cur_similar_reliability <= ver.reliability:
                    result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                              cur_correct_val, cur_correct_val, self._id, self.name,
                                              self.normed_connectivity_matrix, i, ver, None, experiment_name))
                    correct_similar_versions.add(ver)

            similar_base_error_val = cur_correct_val
            if len(correct_similar_versions) == 0:
                similar_base_error_val = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)
            # Границы такие же, как определялись в матрице смежностей для отнесения версий к похожим (строка ~430).
            diversity_coefficient = uniform(0.06, 0.39).__round__(self.round_to)

            for ver in similar_versions:
                if cur_similar_reliability > ver.reliability:
                    similar_error_val = normalvariate(
                        similar_base_error_val, diversity_coefficient * similar_base_error_val
                    ).__round__(self.round_to)
                    result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                              similar_error_val, cur_correct_val, self._id, self.name,
                                              self.normed_connectivity_matrix, i, ver, None, experiment_name))
            # 3. Генерируем выходные данные для частично схожих версий
            # Здесь мы уже генерируем случайное число для каждой из версий, чтобы определить, позволила ли ей её
            # надёжность отработать корректно на данной итерации. Если надёжность позволила, то хорошо, если нет, то
            # сравниваем уровень диверсифицированности версий со случайным числом. Если случайное число превышает
            # уровень диверсифицированности версии, то считаем, что такие мультиверсии зависимы - для них генерируем
            # единый неверный результат, для частично зависимых версии генерируем ответ по нормальному закону
            # распределения, иначе - генерируем независимый ошибочный результат.
            cur_partly_similar_diversity = random()
            partly_similar_depended_versions = set()
            partly_similar_independent_versions = set()
            for ver1 in partly_similar_versions:
                for ver2 in partly_similar_versions:
                    v1_index = self._versions_list.index(ver1)
                    v2_index = self._versions_list.index(ver2)
                    # Если диверсифицировнность версий больше или равна случайному уровню диверсифицированности, то
                    # считаем их различными, иначе - считаем зависимыми
                    if self.normed_connectivity_matrix[v1_index][v2_index] >= cur_partly_similar_diversity:
                        partly_similar_depended_versions.add(ver1)
                        partly_similar_depended_versions.add(ver2)
                    else:
                        partly_similar_independent_versions.add(ver1)
                        partly_similar_independent_versions.add(ver2)

            correct_partly_similar_versions = set()
            for ver in partly_similar_versions:
                cur_partly_similar_reliability = random()
                if cur_partly_similar_reliability <= ver.reliability:
                    result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                              cur_correct_val, cur_correct_val, self._id, self.name,
                                              self.normed_connectivity_matrix, i, ver, None, experiment_name))
                    correct_partly_similar_versions.add(ver)

            error_partly_similar_versions = partly_similar_versions.difference(correct_partly_similar_versions)
            error_partly_abs_depended_versions = error_partly_similar_versions.intersection(
                partly_similar_depended_versions.difference(partly_similar_independent_versions)
            )
            error_partly_abs_independent_versions = error_partly_similar_versions.intersection(
                partly_similar_independent_versions.difference(partly_similar_depended_versions)
            )
            error_partly_similar_versions.intersection_update(
                partly_similar_independent_versions.intersection(partly_similar_depended_versions)
            )

            # Для всех абсолютно схожих версий ставим в соответствие одно неверное значение
            partly_similar_base_error_val = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)
            for ver in error_partly_abs_depended_versions:
                result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                          partly_similar_base_error_val, cur_correct_val, self._id, self.name,
                                          self.normed_connectivity_matrix, i, ver, None, experiment_name))
            # Для частично схожих версий генерирует ответы по нормальному распределению на основе одного значения
            for ver in error_partly_similar_versions:
                partly_similar_error_val = normalvariate(
                    partly_similar_base_error_val, cur_partly_similar_diversity * partly_similar_base_error_val
                ).__round__(self.round_to)
                result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                          partly_similar_error_val, cur_correct_val, self._id, self.name,
                                          self.normed_connectivity_matrix, i, ver, None, experiment_name))
            # Для полностью разных версий генерируем абсолюно независимые ответы
            for ver in error_partly_abs_independent_versions:
                partly_similar_independent_error_val = uniform(self.min_out_val,
                                                               self.max_out_val).__round__(self.round_to)
                result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                          partly_similar_independent_error_val, cur_correct_val, self._id, self.name,
                                          self.normed_connectivity_matrix, i, ver, None, experiment_name))

            # 4. Генерируем выходные данные для явно несхожих версий
            for ver in difference_versions:
                cur_difference_reliability = random()
                cur_dif_version_answer = cur_correct_val
                if cur_difference_reliability > ver.reliability:
                    cur_dif_version_answer = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)

                result_lst.append(NResult(ver.id, ver.name, ver.reliability, ver.common_coordinates_list,
                                          cur_dif_version_answer, cur_correct_val, self._id,
                                          self.name,
                                          self.normed_connectivity_matrix, i, ver, None, experiment_name))
                correct_partly_similar_versions.add(ver)
            # ---------------------------------------------------------------
            self._global_results_lst.append(result_lst)

        self._experiment_name = experiment_name
        self._get_global_results_lst_2_write()
        return self._global_results_lst

    # def generate_experiment_data(self, iterations_amount: int) -> list:
    #     tmp_dependent_set = set()
    #     # Чтобы не возникало неопределённости при записи результатов в БД, очищаем имеющиеся результаты перед генерацией
    #     self._global_results_lst_2_write = list()
    #     self._global_results_lst = list()
    #     for i in range(iterations_amount):
    #         result_lst = []
    #         # Также генерируем число для определения вероятности возникновения зависимых ошибок на текущей итерации
    #         cur_iteration_diversity = random()
    #
    #         cur_correct_val = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)
    #         correct_versions = set()
    #         error_versions = set()
    #         dependent_versions = set()
    #         absolutely_dependent_versions = set()
    #         independent_versions = set()
    #         is_there_absolutely_dependent_versions = True
    #         for j in range(len(self._versions_list)):
    #             # Генерирую число, которое покажет, отраотала ли версия верно. Если оно будет больше надёжности
    #             # мультиверсии, то считаем, что версия выдаёт ошибочное значение
    #             cur_iteration_reliability = random()
    #             # Если текущая версия на данной итерации выполнения модуля выдала верный ответ, записываем его в список
    #             if self._versions_list[j].reliability >= cur_iteration_reliability:
    #                 result_lst.append(NResult(self._versions_list[j].id, self._versions_list[j].name,
    #                                           self._versions_list[j].reliability,
    #                                           self._versions_list[j].common_coordinates_list, cur_correct_val,
    #                                           cur_correct_val, self._id, self.name, self.normed_connectivity_matrix, i,
    #                                           self._versions_list[j]))
    #                 # result_lst.append(cur_result)
    #                 correct_versions.add(self._versions_list[j])
    #             else:  # Если версия выдала неверный результат,
    #                 tmp_dependent_set.clear()
    #                 for k in range(len(self._versions_list)):
    #                     # Если текущая j-я версия и другая версия из вложенного цикла слабо диверсифицированы,
    #                     if self.normed_connectivity_matrix[j][k] < cur_iteration_diversity and j != k:
    #                         # то добавляем их в множество для генерации единого ответа,
    #                         dependent_versions.add(self._versions_list[j])
    #                         dependent_versions.add(self._versions_list[k])
    #
    #                         tmp_dependent_set.add(self._versions_list[j])
    #                         tmp_dependent_set.add(self._versions_list[k])
    #                     else:  # иначе добавляем версии в множество для генерации независимых ответов
    #                         independent_versions.add(self._versions_list[j])
    #                         independent_versions.add(self._versions_list[k])
    #                 # Абсолютно зависимые версии, т.е. те, которые на каждой j-ой итерации будут добавляться в множество
    #                 # зависимых версий, получат одинаковый ошибочный выход. Тут мы их определяем
    #                 if is_there_absolutely_dependent_versions:
    #                     if len(tmp_dependent_set) > 0 and len(absolutely_dependent_versions) > 0:
    #                         absolutely_dependent_versions.intersection_update(tmp_dependent_set)
    #                         # Если на очередной j-й итерации все зависимые элементы обновились, то абсолютно зависимых
    #                         # элементов у нас быть не может, т.к. надо чтобы одинаковые элементы встречались на всех
    #                         # итерациях
    #                         if len(absolutely_dependent_versions) == 0:
    #                             is_there_absolutely_dependent_versions = False
    #                     # На 1-й итерации абсолютно независимых версий ещё нет - заполняем их
    #                     elif len(tmp_dependent_set) > 0 and len(absolutely_dependent_versions) == 0 and j == 0:
    #                         absolutely_dependent_versions = set(tmp_dependent_set)
    #                     else:
    #                         is_there_absolutely_dependent_versions = False
    #                         absolutely_dependent_versions.clear()
    #
    #                 # Добавляем версию в множество, где хранятся версии, выдавшие неверный результат в текущем запуске
    #                 error_versions.add(self._versions_list[j])
    #
    #         # Теперь находим полностью зависимые друг от друга, т.е. расположенные друг к другу блже, чем
    #         # cur_iteration_diversity, и полностью независимые версии.
    #         # absolutely_dependent_versions = dependent_versions.difference(independent_versions)
    #         absolutely_independent_versions = independent_versions.difference(dependent_versions)
    #
    #         # Определяем группу версий, выдавших независимые ошибки и генерируем для них значения независимо
    #         independent_error_versions = absolutely_independent_versions.intersection(error_versions)
    #         temporary_error_answers = list()
    #         for err_v in independent_error_versions:
    #             err_answer = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)
    #             result_lst.append(NResult(err_v.id, err_v.name, err_v.reliability, err_v.common_coordinates_list,
    #                                       err_answer, cur_correct_val, self._id, self.name,
    #                                       self.normed_connectivity_matrix, i, err_v))
    #             temporary_error_answers.append({'version': err_v, 'answer': err_answer})
    #
    #         # Версии, которые входят и в зависимые, и в независимые, будем считать частично зависимыми, т.е. варианты
    #         # ответов для них будут разными, но в пределах нормального распределения от версии, от которой зависят.
    #         partly_depended_versions = dependent_versions.difference(absolutely_dependent_versions)
    #         partly_depended_versions.difference_update(absolutely_independent_versions)
    #         partly_depended_versions.intersection_update(error_versions)
    #
    #         # Определяем, минимаьное расстояние от одной из зависимых до одной из независимых версий. Если оно
    #         # окажется меньше cur_iteration_diversity, то будем генерировтаь значения для зависимых на основе
    #         # значения самой близкой к ним независимой весии
    #         min_distance = float('inf')
    #         min_distance_ver_index = None
    #         for dep_ver in partly_depended_versions:
    #             dep_ver_index = self._versions_list.index(dep_ver)
    #             for n in range(len(self._versions_list)):
    #                 if (self.normed_connectivity_matrix[dep_ver_index][n] < min_distance or min_distance is None) \
    #                         and self._versions_list[n] in absolutely_independent_versions:
    #                     min_distance = self.normed_connectivity_matrix[dep_ver_index][n]
    #                     min_distance_ver_index = n
    #
    #         # Если минимальное расстояние между зависимыми и независимыми версиями меньше cur_iteration_diversity,
    #         # то берём значение независимой версии, до которой минимальное расстояние за базовое для генерации
    #         # других значений по нормальному распределению. Иначе, генерируем базовое число случайно
    #         base_val_depended_on = None
    #         if min_distance >= cur_iteration_diversity or min_distance_ver_index is None:  # min_distance == float('inf') or
    #             base_val_depended_on = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)
    #         else:
    #             # Если минимально близкой версией является версия, выдавшая неверное значение, берём его за базовое
    #             for cur_item in independent_error_versions:
    #                 if self._versions_list[min_distance_ver_index] == cur_item['version']:
    #                     base_val_depended_on = cur_item['answer']
    #                     break
    #             else:  # Если максимальна близка версия с верным ответом, то берём верный ответ за базовые
    #                 base_val_depended_on = cur_correct_val
    #
    #         if base_val_depended_on is not None:
    #             # В одних пределах для всех частично зависимых версий генерируем разные значения
    #             for dep_ver in partly_depended_versions:
    #                 err_answer = normalvariate(base_val_depended_on,
    #                                            cur_iteration_diversity * base_val_depended_on).__round__(self.round_to)
    #                 result_lst.append(NResult(dep_ver.id, dep_ver.name, dep_ver.reliability,
    #                                           dep_ver.common_coordinates_list, err_answer, cur_correct_val, self._id,
    #                                           self.name, self.normed_connectivity_matrix, i, dep_ver))
    #
    #             # Генерируем одно значение для всех абсолютно зависимых версий из нормального распределения от базового
    #             # значения в пределах cur_iteration_diversity от него
    #             abs_err_answer = normalvariate(base_val_depended_on,
    #                                            cur_iteration_diversity * base_val_depended_on).__round__(self.round_to)
    #             for abs_err_ver in absolutely_dependent_versions.intersection(error_versions):
    #                 result_lst.append(NResult(abs_err_ver.id, abs_err_ver.name, abs_err_ver.reliability,
    #                                           abs_err_ver.common_coordinates_list, abs_err_answer, cur_correct_val,
    #                                           self._id, self.name, self.normed_connectivity_matrix, i, abs_err_ver))
    #
    #         self._global_results_lst.append(result_lst)
    #
    #     self._get_global_results_lst_2_write()
    #
    #     return self._global_results_lst

    def save_module(self):
        cur_conn = DBConnector(self._db_name)
        if not cur_conn.table_exists('module'):
            create_query = 'create table module ("id" integer primary key autoincrement not null, '
            create_query += '"name" varchar(255) not null, "round_to" integer not null, '
            create_query += '"dynamic_diversities_intervals" varchar(1024) null, '
            create_query += '"const_diversities_count" integer not null, "dynamic_diversities_count" integer null, '
            create_query += '"min_out_val" real not null, "max_out_val" real null);'
            cur_conn.execute_query(create_query, [], True, False)

        if self._id is None:
            insert_query = f"insert into module (name, round_to, dynamic_diversities_intervals, const_diversities_count"
            insert_query += f", dynamic_diversities_count, min_out_val, max_out_val) values ('{self.name}', "
            insert_query += f"{self.round_to}, '{json.dumps(self._dynamic_diversities_intervals_dict)}', "
            insert_query += f"{self._const_diversities_count}, {self._dynamic_diversities_count}, {self.min_out_val}, "
            insert_query += f'{self.max_out_val});'
            # Т.к. у нас возвращается список кортежей, берём первый элемент первого кортежа, т.к. id сего 1 возвращется!
            self._id = cur_conn.execute_query(insert_query, [], True)[0][0]
        else:
            update_query = f"update module set name = '{self.name}', round_to = {self.round_to}, "
            update_query += f"dynamic_diversities_intervals = '{json.dumps(self._dynamic_diversities_intervals_dict)}',"
            update_query += f"const_diversities_count = '{self._const_diversities_count}', "
            update_query += f"dynamic_diversities_count = {self._dynamic_diversities_count}, "
            update_query += f"min_out_val = {self.min_out_val}, max_out_val = {self.max_out_val} where id = {self._id};"
            cur_conn.execute_query(update_query, [], True, False)

    def save_module_with_versions(self):
        self.save_module()
        for ver in self._versions_list:
            ver.save(self._id)

    def save_experiment_data(self):
        if len(self._global_results_lst_2_write) > 0:
            cur_conn = DBConnector(self._db_name)
            if not cur_conn.table_exists('experiment_data'):
                create_query = 'create table experiment_data ("id" integer primary key autoincrement not null, '
                create_query += 'version_id integer not null, version_name varchar(255), version_reliability real not '
                create_query += 'null, version_common_coordinates varchar(1024) not null, version_answer real not null,'
                create_query += 'correct_answer real not null, module_id integer not null, module_name varchar(255), '
                create_query += 'module_connectivity_matrix varchar(4095), module_iteration_num int not null, '
                create_query += 'experiment_name varchar(31) not null, '
                create_query += 'unique(version_id, version_reliability, version_common_coordinates, version_answer, '
                create_query += 'correct_answer, module_id, module_connectivity_matrix, module_iteration_num) '
                create_query += 'on conflict replace, foreign key ("version_id") references version(id), '
                create_query += 'foreign key ("module_id") references module(id));'
                cur_conn.execute_query(create_query, [], True, False)

            check_experiment_name_query = 'select '
            # Обращаемся к первому результату первой итерации для провеки, присвоен ли ему id, чтобы понять, надо ли
            # сохранять результаты, или они уже сохранены, т.к. можно только перегенерировать их, но не изменить
            if self._global_results_lst[0][0].id is None:
                insert_query = 'insert into experiment_data (version_id, version_name, version_reliability, '
                insert_query += 'version_common_coordinates, version_answer, correct_answer, module_id, module_name, '
                insert_query += 'module_connectivity_matrix, module_iteration_num, experiment_name) '
                insert_query += ' values(?,?,?,?,?,?,?,?,?,?,?);'
                cur_conn.execute_query(insert_query, self._global_results_lst_2_write, True, False)

                if input('Do you want to load IDs of saved data? Yes - Y; No - any key').upper() == 'Y':
                    self.load_experiment_data(self._experiment_name)
                # for iteration in self._global_results_lst:
                #     for cur_res in iteration:
                #         select_query = f"select id from experiment_data where version_id = {cur_res.version_id} and "
                #         select_query += f"version_reliability = {cur_res.version_reliability} and "
                #         select_query += f"version_common_coordinates = "
                #         select_query += f"'{json.dumps({'version_coordinates': cur_res.version_common_coordinates})}' "
                #         select_query += f"and version_answer = '{cur_res.version_answer}' and correct_answer = "
                #         select_query += f"{cur_res.correct_answer} and module_id = {cur_res.module_id} and "
                #         select_query += f"module_connectivity_matrix = "
                #         select_query += f"'{json.dumps({'connectivity_matrix': cur_res.module_connectivity_matrix})}' "
                #         select_query += f"and module_iteration_num = {cur_res.module_iteration_num};"
                #
                #         select_res = cur_conn.execute_query(select_query)
                #         cur_res.id = select_res[0][0]

            # Да, я пониаю, что в цикле выполнять одиночные запросы, как в коде выше, к БД неоптимально, но для
            # однопользовательской базы приемлимо
        else:
            raise LookupError('There is no experiment data to save into data base')

    def load_experiment_data(self, experiment_name: str = None):
        cur_conn = DBConnector(self._db_name)
        if cur_conn.table_exists('experiment_data'):
            can_we_go_further = False
            if experiment_name is None and self._experiment_name is not None:
                experiment_name = self._experiment_name
                can_we_go_further = True

            if experiment_name is not None:
                can_we_go_further = True

            if can_we_go_further:
                select_query = "select id, version_id, version_name, version_reliability, version_common_coordinates, "
                select_query += "version_answer, correct_answer, module_id, module_name, module_connectivity_matrix, "
                select_query += "module_iteration_num, experiment_name from experiment_data where experiment_name = "
                select_query += f"'{experiment_name}';"

                select_res = cur_conn.execute_query(select_query)
                # Если удалось загрузить данные из БД,
                if len(select_res) > 0:
                    # то очищает имеющиеся списки с результатами для загрузки новых данных
                    self._global_results_lst = list()
                    self._global_results_lst_2_write = list()

                cur_iter_num = None
                cur_iter_list = []
                for res in select_res:
                    # Если это не самая первая итерация и итерация эксперимента сменилась, записываем собранные данные в
                    # глобальный список и очищаем временный список
                    if cur_iter_num is not None and cur_iter_num != res[10]:
                        self._global_results_lst.append(cur_iter_list)
                        cur_iter_list = list()

                    cur_iter_num = res[10]
                    cur_iter_list.append(NResult(res[1], res[2], res[3], json.loads(res[4])['version_coordinates'],
                                                 res[5], res[6], res[7], res[8],
                                                 json.loads(res[9])['connectivity_matrix'], res[10],
                                                 self.get_version_by_id(res[1]), res[0], res[11]))
                    if self._experiment_name is None:
                        self._experiment_name = res[11]

                if len(select_res) > 0:
                    # Дозаписываем данные в глобальный массив результатов с последней итерации эксперимента
                    self._global_results_lst.append(cur_iter_list)

                self._get_global_results_lst_2_write()
            else:
                raise ValueError(
                    f'Unexpected value {experiment_name} of experiment_name parameter!'
                )
        else:
            raise LookupError(
                f'There is no "EXPERIMENT_DATA" table in {self._db_name} data base. Save experiment data before load it'
            )

    def get_experiments_names(self):
        cur_conn = DBConnector(self._db_name)
        if cur_conn.table_exists('experiment_data'):
            experiment_select_res = cur_conn.execute_query("select distinct experiment_name from experiment_data;")
            return [exp_name[0] for exp_name in experiment_select_res]

    # def load_experiment_data(self):
    #     cur_conn = DBConnector(self._db_name)
    #     if cur_conn.table_exists('experiment_data'):
    #         select_query = f"select id, version_id, version_name, version_reliability, version_common_coordinates, "
    #         select_query += f"version_answer, correct_answer, module_id, module_name, module_connectivity_matrix, "
    #         select_query += f"module_iteration_num, experiment_name from experiment_data where module_id = {self._id}"
    #         select_query += f" and module_connectivity_matrix = "
    #         select_query += f"'{json.dumps({'connectivity_matrix': self.normed_connectivity_matrix})}' "
    #         select_where_query = " and ("
    #         versions_counter = 0
    #         for version in self._versions_list:
    #             if versions_counter > 0:
    #                 select_where_query += " or "
    #             select_where_query += f"(version_id = {version.id} and version_reliability = {version.reliability})"
    #             select_where_query += f"(version_common_coordinates = "
    #             select_where_query += f"'{json.dumps({'version_coordinates': version.common_coordinates_list})}'"
    #             versions_counter += 1
    #         select_where_query += ");"
    #
    #         select_res = cur_conn.execute_query(select_query + select_where_query)
    #         # Если удалось загрузить данные из БД, то очищает имеющиеся списки с результатами для загрузки новых данных
    #         if len(select_res) > 0:
    #             self._global_results_lst = list()
    #             self._global_results_lst_2_write = list()
    #
    #         cur_iter_num = None
    #         cur_iter_list = []
    #         for res in select_res:
    #             # Если это не самая первая итерация и итерация эксперимента сменилась, записываем собранные данные в
    #             # глобальный список и очищаем временный список
    #             if cur_iter_num is not None and cur_iter_num != res[10]:
    #                 self._global_results_lst.append(cur_iter_list)
    #                 cur_iter_list = list()
    #
    #             cur_iter_num = res[10]
    #             cur_iter_list.append(NResult(res[1], res[2], res[3], json.loads(res[4])['version_coordinates'], res[5],
    #                                          res[6], res[7], res[8], json.loads(res[9])['connectivity_matrix'], res[10],
    #                                          None, res[0]))
    #         if len(select_res) > 0:
    #             # Дозаписываем данные в глобальный массив результатов с последней итерации эксперимента
    #             self._global_results_lst.append(cur_iter_list)
    #
    #         self._get_global_results_lst_2_write()
    #     else:
    #         raise LookupError(
    #             f'There is no "EXPERIMENT_DATA" table in {self._db_name} data base. Save experiment data before load it'
    #         )

    def load_module(self):
        cur_conn = DBConnector(self._db_name)
        if cur_conn.table_exists('module'):
            if self._id is None:
                q_set = cur_conn.execute_query('select distinct id, name, round_to from module order by id;')
                get_num_str = 'Choice module by id to load it.\n'
                for mdl in q_set:
                    get_num_str += f'\n{str(mdl)}'
                get_num_str += '\n'
                chosen_id = input_num(get_num_str)
            else:
                chosen_id = self._id

            select_query = f'select id, name, round_to, dynamic_diversities_intervals, const_diversities_count, '
            select_query += f'dynamic_diversities_count, min_out_val, max_out_val from module where id = {chosen_id};'
            select_res = cur_conn.execute_query(select_query)

            # Т.к. у нас только 1 модуль может быть найден по id, то обращаемся мы к 0-му индексу, без доп. проверок
            self._id = select_res[0][0]
            self.name = select_res[0][1]
            self.round_to = select_res[0][2]
            self._dynamic_diversities_intervals_dict = json.loads(select_res[0][3])
            self._const_diversities_count = select_res[0][4]
            self._dynamic_diversities_count = select_res[0][5]
            self.min_out_val = select_res[0][6]
            self.max_out_val = select_res[0][7]
        else:
            raise LookupError(
                f'There is no "MODULE" table in {self._db_name} data base. Save module data before load it.'
            )

    def load_module_with_versions(self):
        try:
            self.load_module()
            self._versions_list = NVersion.load_versions_2_module(self._id)
        except LookupError as e:
            print(str(e))
        except AttributeError as e:
            print(str(e))

    # def load_experiment_data(self):
    #     cur_conn = DBConnector(self._db_name)
    #     if cur_conn.table_exists('experiment_data'):
    #         if self._id is not None:
    #             select_query = f'select id, name, round_to, dynamic_diversities_intervals, const_diversities_count, '
    #             select_query += f'dynamic_diversities_count, min_out_val, max_out_val from module where id = {chosen_id};'
    #             select_res = cur_conn.execute_query(select_query)
    #
    #             # Т.к. у нас только 1 модуль может быть найден по id, то обращаемся мы к 0-му индексу, без доп. проверок
    #             self._id = select_res[0][0]
    #             self.name = select_res[0][1]
    #             self.round_to = select_res[0][2]
    #             self._dynamic_diversities_intervals_dict = json.loads(select_res[0][3])
    #             self._const_diversities_count = select_res[0][4]
    #             self._dynamic_diversities_count = select_res[0][5]
    #             self.min_out_val = select_res[0][6]
    #             self.max_out_val = select_res[0][7]
    #     else:
    #         raise LookupError(
    #             f'There is no "MODULE" table in {self._db_name} data base. Save module data before load it.'
    #         )

    def __str__(self):
        res_str = f'id: {self._id}\tname: {self.name}\tround to: {self.round_to}\tdynamic diversities intervals: '
        res_str += f'{self._dynamic_diversities_intervals_dict}\tconst diversities count: '
        res_str += f'{self._const_diversities_count}\tdynamic diversities count: {self._dynamic_diversities_count}'
        return res_str
