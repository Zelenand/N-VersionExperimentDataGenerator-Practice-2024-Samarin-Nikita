"""
t/(n-1) vote algorithm programming realization
https://eprints.ncl.ac.uk/file_store/production/59232/8480AF9B-598A-462C-8417-1B1F8CAA6190.pdf

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""
from nversionmodule.data_generator import NResult


def vote(results: list[NResult]) -> float:
    # ��������� �� ������� �� �����, ������� ������ ����� �������� � �������� (� ������� - ����������� t), ���������� ��
    # ���������� ��� ����������� ��������� ��� ������� ������ (n - ����� ���������� �����������) �� �������: n >= 2t + 1
    # � ������ ������ (����� ������������ ������ t ������������ ����� ��� �������� n), ��� ������� ���������� ����������
    n = len(results)
    t = ((n - 1) / 2).__trunc__()  # ����������� ������� ����� �� ������ ������ � ������ ����������� ������

    # �.�. �� ����� � ���������� ���� �� ��� ������, ���� ��������� �� �������� ��� �������� ���� ������������ ������ �
    # ���������� ������������ ��������� ������
    output_count = n - t

    # � ������, ����� t �����������, ��������� ������� �� ������������ � ����������, � ��� �������� ������������ �
    # �������� ���������. ����� ���� �� ����� ������ ���������� ������ � ��������� ������������ ������, � ����� ��������
    # ������ �� ������ ������������ ������ (n - 1), ������� � ������� � ����� (n - 1) / (output_count - 1). ��
    # output_count ���������� 1, �.�. ���� ����� ����� ����� ����������� �� ������ � ��������� ��������
    output_indexes_step = ((n - 1) / (output_count - 1)).__trunc__()
    output_indexes = {i: results[i].version_answer for i in range(0, n - 1, output_indexes_step)}
    output_indexes[n - 1] = results[n - 1].version_answer

    comparators = dict()
    max_count_indexes = dict()
    cur_start_index = -1
    max_group_count = 0  # ������������ ����� ������� ���������� ������
    cur_max_group_count = 0  # ������� ����� ������ � ���������� �������
    # �������� 2, �.�. ��������� ������� �� ��������� � ���������, � � ����� ������������ i + 1 �������
    for i in range(n - 2):
        if results[i].version_answer == results[i + 1].version_answer:
            comparators[(i, i + 1)] = 0

            # ����� ������� ������������ ����� ������ � ���������� �������, ����� �� �������� �������� � ��������� ����
            if cur_max_group_count == 0:
                cur_start_index = i
                # ���� ������ ������� � ������ ���� ������ �� ������, �� ����� ����������� 2, ����� ��������� �� �������
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

    # ��������� ����� ���� ��� �� ������, ����� ��������� ������ ������������ ������ �������� ����������, � � else ����
    # ����� ������ �� ���� ����������� ����� ����������� �����
    if cur_start_index != -1:
        if cur_max_group_count not in max_count_indexes:
            max_count_indexes[cur_max_group_count] = [(cur_start_index, n - 2)]
        else:
            max_count_indexes[cur_max_group_count].append((cur_start_index, n - 2))

    # ���� ������������ ����� ������ � ���������� ������� = t, �� ��� ������ - �������� �� ������ �����, (��������
    # ������, ���� ������ ���� ��������� ��������� ����� �������������), ���� ������� ���� ������ ����������� �������
    # > t, �� ��� ����� ������ �����, ���� ������� 2 ���������� ����������� ������ >= t, ��� ����� �� ������ �� ����
    # ���������� ��������, �� ���� �������� ��������� ������ (�� �������������� � ���������)
    correct_result: float
    # ���� ������� ������ ������ ������ ������, ������ �� ����� �� ������ ����� ������, ������� ����� ������ ������,
    if max_group_count >= t:
        # �� ���������, ����� �� ���� ������ ����� ������ ���������� (��� ������ > t ���������, ����� �������� � �����
        # ������������ ���������� �������: n > 2t + 1, �.�. ����� ���� ��������� ������)
        if len(max_count_indexes[max_group_count]) == 1:
            # ���� ���������� ���� ���� ����� ������ ������, �� ������� �� ����� ������. ��� ����� ������� ������ ������
            # ������ �� ����� ���������, ��� ����� ����� �� �����
            for key, val in output_indexes.items():
                if max_count_indexes[max_group_count][0][0] <= key <= max_count_indexes[max_group_count][0][1]:
                    correct_result = val
                    break
            else:
                # ���� �� �����-�� ������� ����� �� ��� ������ (���� ����� ����-�� ��������), �� ���� ��� �� �� ������,
                # ��� ����� ������ ���� �� �����, � �� ������, ��� ������ ������ � ���������
                correct_result = results[max_count_indexes[max_group_count][0][0]].version_answer
        else:  # ���� ����� � ������������ ������ ������ > 1, �� �� ����, ����� ������� - �������� ��������� ���������
            correct_result = output_indexes[n - 1]
    else:
        correct_result = output_indexes[n - 1]

    return correct_result
