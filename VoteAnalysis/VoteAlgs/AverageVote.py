from statistics import mean

from nversionmodule.data_generator import NResult


def vote(results: list[NResult]) -> float:
    res_lst = [res.version_answer for res in results]
    return mean(res_lst)
