import collections
from typing import Union, Tuple, Iterable


def pair(x: Union[int, Iterable[int]]) -> Tuple[int, int]:
    """
    If input is iterable (e.g., list or tuple) of length 2, return it as tuple. If input is a single integer, duplicate
    it and return as a tuple.

    Arguments:
    x: Either an iterable of length 2 or a single integer.

    Returns:
    A tuple of length 2.
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))
