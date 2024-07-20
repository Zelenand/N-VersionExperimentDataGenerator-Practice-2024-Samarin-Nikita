import random

from nversionmodule.data_generator import NResult

error_string = "Input list is empty"


def classic_vote(results: list[NResult], return_classes_list: bool = False) -> float:
    classes_list: list[list] = []
    for res in results:
        # Если другие мультиверсии уже выставили свои ответы на голосование, то сравниваем текущий ответ с ними
        if len(classes_list) > 0:
            # Если обрабатываемый ответ относится к уже имеющемуся классу, то добавляем его в этот класс.
            for answers_class in classes_list:
                if res.version_answer == answers_class[0].version_answer:
                    answers_class.append(res)
                    break
            else:  # Иначе создаём новый класс для ответа, т.к. он встретился впервые
                classes_list.append([res])
        else:
            classes_list.append([res])

    # Если есть из чего выбирать, выбираем первую группу, которая имеет наибольшее число одинаковых ответов
    if len(classes_list) > 0:
        if return_classes_list:
            return classes_list
        else:
            max_index = 0
            max_values_in_class = len(classes_list[0])
            max_index_lst = list()
            for i in range(len(classes_list)):
                if 0 < len(classes_list[i]) > max_values_in_class:
                    max_values_in_class = len(classes_list[i])
                    max_index = i
                    max_index_lst.clear()
                    max_index_lst.append(max_index)
                elif len(classes_list[i]) == max_values_in_class != 0:
                    max_index_lst.append(i)

            # Если есть 2 или более класса с одинаковым количеством ответов, то выбираем из них случайным образом
            if len(max_index_lst) > 1:
                max_index = random.choice(max_index_lst)
            # Т.к. во вложенном списке хранятся одинаковые результаты, мы можем возвращать любой, например 0-й
            return classes_list[max_index][0].version_answer
    else:
        print(error_string)
        raise ValueError(error_string)


def calc_versions_diversity(results_list: list[NResult]) -> float:
    max_diversity: float = 0
    if len(results_list) == 1:
        # Если в списке всего одно значение, то формально - это множество из 1-го элемента, он сам от себя не
        # отличается, поэтому можно было бы считать данную группу недеверсифицированной,
        diversity = 0
        # # но всё же надо сравнивать и такие группы, поэтому будем брать их удалённость от начала координат
        # diversity = results_list[0].version.distance_from_zero_point
    elif len(results_list) > 1:
        for j in range(len(results_list)):
            for k in range(j + 1, len(results_list)):
                cur_diversity = results_list[j].version.calculate_distance_to(results_list[k].version)
                if max_diversity < cur_diversity:
                    max_diversity = cur_diversity
        diversity = max_diversity
    else:
        print(error_string)
        raise ValueError(error_string)
    return diversity


def modified_vote(results: list[NResult]) -> float:
    try:
        classes_list = classic_vote(results, True)
    except ValueError as err:
        print(str(err))
        raise ValueError(str(err))
    else:
        max_index = 0
        max_values_in_class = len(classes_list[0])
        max_length = calc_versions_diversity(classes_list[0])
        max_indexes_lst = list()
        for i in range(len(classes_list)):
            if 0 < len(classes_list[i]):
                tmp_max_len = calc_versions_diversity(classes_list[i])
                # Если в одной группе ответов больше чем в другой, однозначно определяем группу (класс) с верным ответом
                if len(classes_list[i]) > max_values_in_class:
                    max_values_in_class = len(classes_list[i])
                    max_index = i
                    max_length = tmp_max_len

                    max_indexes_lst.clear()
                    max_indexes_lst.append(max_index)
                elif len(classes_list[i]) == max_values_in_class:  # Если 2 класса содержат одинаковое число ответов,
                    # то пробуем посмотреть на уровень диверсифицированности этих групп - выбираем ту, где версии более
                    # различны между собой
                    if tmp_max_len > max_length:
                        max_values_in_class = len(classes_list[i])
                        max_index = i
                        max_length = tmp_max_len

                        max_indexes_lst.clear()
                        max_indexes_lst.append(max_index)
                    elif tmp_max_len == max_length:  # Если и число версий в классе, и диверсифицированность версий
                        # внутри классов одинаковые, то возвращаемся к классическому методу - выбираем случайно, т.к.
                        # нет дополнительной информации
                        max_indexes_lst.append(i)

        # Если есть 2 или более класса с одинаковым количеством ответов и одинаковым уровнем диверсифицированности, то
        # выбираем из них случайным образом
        if len(max_indexes_lst) > 1:
            max_index = random.choice(max_indexes_lst)
        # Т.к. во вложенном списке хранятся одинаковые результаты, мы можем возвращать любой, например 0-й
        return classes_list[max_index][0].version_answer
