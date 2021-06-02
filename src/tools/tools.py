# -*- coding: utf-8 -*-


def lazyreadonlyproperty(func):
    """decorator to generate a read-only lazily evaluated property

    Args:
        func: function to be decorated as property

    Returns:
        decorated property
    """

    name = "_lazy_" + func.__name__

    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value

    return lazy
