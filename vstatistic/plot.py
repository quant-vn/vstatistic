try:
    from pandas.plotting import register_matplotlib_converters as _rmc

    _rmc()
except ImportError:
    pass

from .plotting.wrappers import *  # noqa
