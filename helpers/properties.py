"""Helpers for properties and lazy evaluation"""


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated.

    Args:
        fn (function): function that should be lazily evaluated

    Returns:
        result, either calculated or from cache

    Inspired by:
        https://stevenloria.com/lazy-properties/
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
