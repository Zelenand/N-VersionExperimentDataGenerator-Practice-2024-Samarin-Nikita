"""
t/(n-1) vote algorithm programming realization
https://eprints.ncl.ac.uk/file_store/production/59232/8480AF9B-598A-462C-8417-1B1F8CAA6190.pdf

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""
from nversionmodule.data_generator import NResult


def vote(results: list[NResult]) -> float:
    # Поскольку мы заранее не знаем, сколько версий может работать с ошибками (в формуле - переменнная t), определяем их
    # количество как максимально возможное для данного случая (n - общее количество результатов) по формуле: n >= 2t + 1
    # В худшем случае (когда нестабильных версий t максимальное число при заданном n), эта формула становится равенством
    n = len(results)
    t = ((n - 1) / 2).__trunc__()  # Отбрасываем дробную часть на случай работы с чётным количеством версий

    # Т.к. на выход в компаратор идут не все версии, надо вычислить их значение как разность всех сравниваемых версий и
    # количества потенциально ненадёжных версий
    output_count = n - t

    # В слечае, когда t максимальна, последний элемент не сравнивается с остальными, а его значение используется в
    # качестве выходного. Кроме того на выход пойдут результаты первой и последней сравниваемой версий, а также значения
    # версий из набора сравниваемых версий (n - 1), начиная с первого с шагом (n - 1) / (output_count - 1). От
    # output_count отнимается 1, т.к. один выход точно будет приходиться на версию с последним индексом
    output_indexes_step = ((n - 1) / (output_count - 1)).__trunc__()
    output_indexes = {i: results[i].version_answer for i in range(0, n - 1, output_indexes_step)}
    output_indexes[n - 1] = results[n - 1].version_answer

    comparators = dict()
    max_count_indexes = dict()
    cur_start_index = -1
    max_group_count = 0  # Максимальное число версийс одинаковым отвеом
    cur_max_group_count = 0  # Текущее число версий с одинаковым ответом
    # Отнимаем 2, т.к. последний элемент не участвует в сравнении, а в цикле используется i + 1 элемент
    for i in range(n - 2):
        if results[i].version_answer == results[i + 1].version_answer:
            comparators[(i, i + 1)] = 0

            # Сразу считаем максимальное число версий с одинаковым ответом, чтобы не пришлось заходить в следующий цикл
            if cur_max_group_count == 0:
                cur_start_index = i
                # Если ответы совпали у первых двух версий из группы, то сразу присваиваем 2, потом добавляем по единице
                cur_max_group_count = 2
            else:
                cur_max_group_count += 1
        else:
            comparators[(i, i + 1)] = 1

            if cur_start_index != -1:
                if cur_max_group_count not in max_count_indexes:
                    max_count_indexes[cur_max_group_count] = [(cur_start_index, i)]
                else:
                    max_count_indexes[cur_max_group_count].append((cur_start_index, i))

            cur_start_index = -1
            cur_max_group_count = 0

        if cur_max_group_count > max_group_count:
            max_group_count = cur_max_group_count

    # Дублируем здесь этот код на случай, когда последняя группа сравниваемых версий является правильной, и в else выше
    # зайти просто не было возможности перед завершением цикла
    if cur_start_index != -1:
        if cur_max_group_count not in max_count_indexes:
            max_count_indexes[cur_max_group_count] = [(cur_start_index, n - 2)]
        else:
            max_count_indexes[cur_max_group_count].append((cur_start_index, n - 2))

    # Если максимальное число версий с одинаковым ответом = t, то эти версии - кандидат на верный ответ, (окажется
    # верным, если ответы всех остальных сравнений будут несогласованы), если найдена одна группа совпадающих ответов
    # > t, то она выдаёт верный ответ, если найдены 2 одинаковые совпадающие группы >= t, или ответ не найден по двум
    # предыдущим правилам, то берём значение последней версии (не учавствовавшей в сравнении)
    correct_result: float
    # Если совпали ответы подряд идущих версий, причём их число не меньше числа версий, которые могут выдать ошибку,
    if max_group_count >= t:
        # то проверяем, всего ли одна группа таких версий существует (для случая > t проверяем, чтобы работало и когда
        # нераввенство становится строгим: n > 2t + 1, т.е. когда мало ошибочных версий)
        if len(max_count_indexes[max_group_count]) == 1:
            # Если существует лишь одна такая группа версий, то считаем их ответ верным. Для этого находим индекс первой
            # версии из этого диапазона, чей ответ пойдёт на выход
            for key, val in output_indexes.items():
                if max_count_indexes[max_group_count][0][0] <= key <= max_count_indexes[max_group_count][0][1]:
                    correct_result = val
                    break
            else:
                # Если по какой-то причине ответ не был найден (хотя такое вряд-ли возможно), то берём его не от версии,
                # чей ответ должен идти на выход, а от версии, чей индекс первый в диапазоне
                correct_result = results[max_count_indexes[max_group_count][0][0]].version_answer
        else:  # Если групп с максимальным числом версий > 1, то не ясно, какую выбрать - выбираем последний результат
            correct_result = output_indexes[n - 1]
    else:
        correct_result = output_indexes[n - 1]

    return correct_result
