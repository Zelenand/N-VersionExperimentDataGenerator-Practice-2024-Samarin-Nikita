"""Experiment data generation module to simulate N-version programming module running

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""
import json
from math import sqrt
from random import random, uniform, normalvariate

from dataaccess import versions_module_data_access
from userinput.user_input import input_num

__author__ = "Denis V. Gruzenkin"
__copyright__ = "Copyright 2021, Denis V. Gruzenkin"
__credits__ = ["Denis V. Gruzenkin"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Denis V. Gruzenkin"
__email__ = "gruzenkin.denis@good-look.su"
__status__ = "Production"


class NVersion:
    """
    Real NVP version emulation
    """
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

    def generate_dynamic_diversities(self, intervals_lst: list[tuple], round_to=6):
        for tpl in intervals_lst:
            self._dynamic_diversities.append(uniform(tpl[0], tpl[1]).__round__(round_to))

    def calculate_distance_to(self, another_version) -> float:
        return self._calc_euclidean_distance(self.common_coordinates_list, another_version.common_coordinates_list)

    def __str__(self):
        return f'{self._id}. {self._name} [{self._reliability}] {self.common_coordinates_list}'

    def save(self, module_id=None):
        """Save version to DB"""
        result = versions_module_data_access.nversion_save(self, module_id)
        if result is not None:
            self._id = result

    def load(self):
        """Load version from DB"""
        select_res = versions_module_data_access.nversion_load(self)

        # Т.к. у нас только 1 версия может быть найден по id, то обращаемся мы к 0-му индексу, без доп. проверок
        self._id = select_res[0][0]
        self.name = select_res[0][1]
        self._const_diversities = json.loads(select_res[0][2])[self.formal_json_db_lst_name]
        self._dynamic_diversities = json.loads(select_res[0][3])[self.formal_json_db_lst_name]
        self._reliability = float(select_res[0][4])

    @classmethod
    def load_versions_to_module(cls, module_id: int):
        """Load several versions of certain module from DB"""
        select_res = versions_module_data_access.nversion_load_versions_2_module(module_id)

        result_list = []
        print(select_res)
        for res in select_res:
            tmp_ver = NVersion(res[0], res[1], json.loads(res[2])[cls.formal_json_db_lst_name],
                               json.loads(res[3])[cls.formal_json_db_lst_name], float(res[4]))
            result_list.append(tmp_ver)
        return result_list


class NResult:
    """Vote algorithm's work result"""

    def __init__(self, v_id: int, v_name: str, v_reliability: float, v_common_coordinates: tuple, v_answer: float
                 , c_answer: float, m_id: int, m_name: str, m_connectivity_matrix: tuple, m_iteration_num: int,
                 version: NVersion = None, res_id=None, experiment_name: str = 'NoName'):
        """NResult constructor
        :param v_id: version ID
        :param v_name: version name
        :param v_reliability: version reliability
        :param v_common_coordinates: common coordinates
        :param v_answer: version answer
        :param c_answer: correct answer
        :param m_id: module ID
        :param m_name: module name
        :param m_connectivity_matrix: module connectivity matrix
        :param m_iteration_num: module iteration number
        :param version: version
        :param res_id: result ID
        :param experiment_name: experiment name"""
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

            max_val = max(max(row) for row in matrix)
            try:
                for i in range(len(matrix)):
                    for j in range(len(matrix[i])):
                        matrix[i][j] /= max_val
            except ZeroDivisionError:
                print('Max distance between all versions is zero, so connectivity matrix cannot by normed')

            return tuple(tuple(row) for row in matrix)
        else:
            raise AttributeError('There are not versions in module')

    @property
    def global_results_lst(self):
        return self._global_results_lst

    def _get_global_results_lst_to_write(self):
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
    def global_results_lst_to_write(self):
        if len(self._global_results_lst_2_write) == 0:
            self._get_global_results_lst_to_write()
        return self._global_results_lst_2_write

    def add_versions(self):
        """Adding version to module"""
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
        """Generate experiment data for module versions"""
        # Сначала находим версии-клоны - для них генерируем одно число для определения надёжности, если число >
        # минимальной надёжности, то версии выдают единый неверный результат, иначе - все выдают верный результат. Берём
        # версии, которые отличаются мало друг от друга и не входят в множество клонов, для них генерируем единое число
        # для определения надёжности, если среди них есть версии, выдавшие корректный результат, то берём этот результат
        # за базовый, иначе - генерируем базовый результат, вблизи которого ренерируем по нормальному распределению
        # ответы версий из этой группы. Находи группу версий, которы отличаются друг от друга примерно на 50%, для них
        # генерируем случайное число 0 или 1, если 0, то действем, как для предыдущей группы, иначе - как для следующей;
        # при этом генерируем число для диверсифицированности, чтобы определить,какие версии получат независимые ответы,
        # а какие сгенерированные по нормальному закону. Если версии различны > 50% примерно, то для каждой из них
        # генерируем разные вероятности на надёжностей и генерируем независимые ошибочные результаты

        # Чтобы не возникало неопределённости при записи результатов в БД, очищаем имеющиеся результаты перед генерацией
        self._global_results_lst_2_write = list()
        self._global_results_lst = list()
        # Запускаем цикл по количеству зананных итераций
        for i in range(iterations_amount):
            result_lst = list()
            # Разбиваем версии на группы для различной генерации результатов их работы
            clone_versions, difference_versions, partly_similar_versions, similar_versions = self._group_versions()
            # Генерируем значение, которое будет считаться правильным ответом на текущей итерации
            cur_correct_val = uniform(self.min_out_val, self.max_out_val).__round__(self.round_to)

            # Теперь для каждой группы версий генерируем результаты их работы
            # ---------------------------------------------------------------
            # 1. Генерируем выходные данные для версий-клонов
            self._clone_versions_data_generation(clone_versions, cur_correct_val, experiment_name, i, result_lst)
            # 2. Генерируем выходные данные для явно схожих версий
            self._similar_versions_data_generation(cur_correct_val, experiment_name, i, result_lst, similar_versions)
            # 3. Генерируем выходные данные для частично схожих версий
            correct_partly_similar_versions = self._partly_similar_versions_data_generation(cur_correct_val,
                                                                                            experiment_name, i,
                                                                                            partly_similar_versions,
                                                                                            result_lst)

            # 4. Генерируем выходные данные для явно несхожих версий
            self._different_versions_data_generation(correct_partly_similar_versions, cur_correct_val,
                                                     difference_versions, experiment_name, i, result_lst)
            # ---------------------------------------------------------------
            self._global_results_lst.append(result_lst)

        self._experiment_name = experiment_name
        self._get_global_results_lst_to_write()
        return self._global_results_lst

    def _clone_versions_data_generation(self, clone_versions, cur_correct_val, experiment_name, i, result_lst):
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

    def _similar_versions_data_generation(self, cur_correct_val, experiment_name, i, result_lst, similar_versions):
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

    def _different_versions_data_generation(self, correct_partly_similar_versions, cur_correct_val, difference_versions,
                                            experiment_name, i, result_lst):
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

    def _partly_similar_versions_data_generation(self, cur_correct_val, experiment_name, i, partly_similar_versions,
                                                 result_lst):
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
        return correct_partly_similar_versions

    def _group_versions(self):
        clone_versions = set()
        similar_versions = set()
        partly_similar_versions = set()
        difference_versions = set()
        for j in range(len(self._versions_list)):
            for k in range(len(self._versions_list)):
                if k > j:  # Сравниваем версии только если они разыне и ещё не сравнивались
                    # Если версии отличаются не более чем на 5%, то считаем их клонами

                    if self._versions_list[k] not in difference_versions and \
                            self._versions_list[k] not in similar_versions and \
                            self._versions_list[k] not in partly_similar_versions:
                        if 0 <= self.normed_connectivity_matrix[j][k] <= 0.05:
                            clone_versions.add(self._versions_list[k])
                        elif 0.05 < self.normed_connectivity_matrix[j][k] < 0.4:  # Если версии похожи, но не клоны
                            similar_versions.add(self._versions_list[k])
                        elif 0.4 <= self.normed_connectivity_matrix[j][
                            k] <= 0.6:  # Если версии частично похожи (~50%)
                            partly_similar_versions.add(self._versions_list[k])
                        else:  # Если версии имеют значительные различия
                            difference_versions.add(self._versions_list[k])
                    if self._versions_list[j] not in difference_versions and \
                            self._versions_list[j] not in similar_versions and \
                            self._versions_list[j] not in partly_similar_versions:
                        if 0 <= self.normed_connectivity_matrix[j][k] <= 0.05:
                            clone_versions.add(self._versions_list[j])
                        elif 0.05 < self.normed_connectivity_matrix[j][k] < 0.4:  # Если версии похожи, но не клоны
                            similar_versions.add(self._versions_list[j])
                        elif 0.4 <= self.normed_connectivity_matrix[j][
                            k] <= 0.6:  # Если версии частично похожи (~50%)
                            partly_similar_versions.add(self._versions_list[j])
                        else:  # Если версии имеют значительные различия
                            difference_versions.add(self._versions_list[j])
        return clone_versions, difference_versions, partly_similar_versions, similar_versions

    def save_module(self):
        """Save module to DB"""
        result = versions_module_data_access.nmodule_save(self)
        if result is not None:
            self._id = result

    def save_module_with_versions(self):
        """Save module and it's versions to DB"""
        self.save_module()
        for ver in self._versions_list:
            ver.save(self._id)

    def save_experiment_data(self):
        """Save experiment data to DB"""
        versions_module_data_access.nmodule_save_experiment_data(self)
        if input('Do you want to load IDs of saved data? Yes - Y; No - any key').upper() == 'Y':
            self.load_experiment_data(self._experiment_name)

    def load_experiment_data(self, experiment_name: str = None):
        """Load experiment data from DB"""
        select_res = versions_module_data_access.nmodule_load_experiment_data(self, experiment_name)
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

        self._get_global_results_lst_to_write()

    @staticmethod
    def get_experiments_names():
        """Load experiments names from database"""
        return versions_module_data_access.get_experiments_names()

    def load_module(self):
        """Load module from DB"""
        select_res = versions_module_data_access.nmodule_load_module(self)

        # Т.к. у нас только 1 модуль может быть найден по id, то обращаемся мы к 0-му индексу, без доп. проверок
        self._id = select_res[0][0]
        self.name = select_res[0][1]
        self.round_to = select_res[0][2]
        self._dynamic_diversities_intervals_dict = json.loads(select_res[0][3])
        self._const_diversities_count = select_res[0][4]
        self._dynamic_diversities_count = select_res[0][5]
        self.min_out_val = select_res[0][6]
        self.max_out_val = select_res[0][7]

    def load_module_with_versions(self):
        """Load module and it's versions from DB'"""
        try:
            self.load_module()
            self._versions_list = NVersion.load_versions_to_module(self._id)
        except LookupError as e:
            print(str(e))
        except AttributeError as e:
            print(str(e))

    def __str__(self):
        res_str = f'id: {self._id}\tname: {self.name}\tround to: {self.round_to}\tdynamic diversities intervals: '
        res_str += f'{self._dynamic_diversities_intervals_dict}\tconst diversities count: '
        res_str += f'{self._const_diversities_count}\tdynamic diversities count: {self._dynamic_diversities_count}'
        return res_str
