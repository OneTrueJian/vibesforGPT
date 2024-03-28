"""helpers for lists and tuples"""
from itertools import groupby


def reduce_list(obj, reduce=True):
    """reduce a with duplicate entries and return tuples of (count, entry)"""
    if reduce:
        return tuple((len(list(g)), k) for k, g in groupby(obj))
    else:
        return obj


def expand_list(obj):
    """expand a list of tuples (count, entry) as produced ty `reduce_list`"""
    if isinstance(obj[0], type(obj)):
        lis = []
        for l in (int(l) * [g] for (l, g) in obj):
            lis.extend(l)
        return lis
    return obj


def list_dim(a: list) -> int:
    """dimension of a (nested) pure Python list, similar to np.shape"""
    if not type(a) == list:
        return []
    if a == []:
        return 0
    return [len(a)] + list_dim(a[0])


def list2str(lis: list) -> str:
    """convert list to string"""
    return "[{}]".format(", ".join([str(el) for el in lis]))
