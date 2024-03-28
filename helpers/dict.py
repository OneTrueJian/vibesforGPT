""" provides AttributeDict """

from collections import OrderedDict


class AttributeDict(OrderedDict):
    """ Ordered dictionary with attribute access """

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"Attribute {attr} not in dictionary, return None.")

    def __dict__(self):
        return self.to_dict()

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        """ (recursively) return plain python dictionary """
        rep = {}
        for key, val in self.items():
            if isinstance(val, AttributeDict):
                val = val.to_dict()
            rep.update({key: val})

        return rep

    def as_dict(self):
        """ return plain python dictionary (Fireworks compatibility) """
        return dict(self)


def merge(source: dict, destination: dict, dict_type=dict) -> dict:
    """recursively merge two dictionaries

    Example:
        a = {"first": {"all_rows": {"pass": "dog", "number": "1"}}}
        b = {"first": {"all_rows": {"fail": "cat", "number": "5"}}}
        merge(b, a) == {
            "first": {"all_rows": {"pass": "dog", "fail": "cat", "number": "5"}}
        }

    Reference:
        https://stackoverflow.com/a/20666342/5172579

    """
    for key, value in source.items():
        if issubclass(value.__class__, dict):
            # get node or create one
            node = destination.setdefault(key, dict_type())
            merge(value, node, dict_type=dict_type)
        else:
            destination[key] = value

    return destination
