import collections


# Avalanche
F_max = 10
F_window = 50
window_factor = 1.0
filter_threshold = 0.1
filter_prominence = 0.2

# file formats
_dct = {"geometry": "aims"}
format = collections.namedtuple("format", _dct.keys())(**_dct)
