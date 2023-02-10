from typing import Tuple, Union

from ._rust import TorchAxisIndex

TorchIndex = Union[Tuple[TorchAxisIndex, ...], TorchAxisIndex]


class Indexer:
    """Helper for defining slices more easily which always returns tuples
    (instead of sometimes returning just TorchAxisIndex).

    Also wraps slices so that they are valid jax types.

    Like interp Indexer, but with torch typing.
    """

    def __getitem__(self, idx: TorchIndex) -> Tuple[TorchAxisIndex, ...]:
        if isinstance(idx, tuple):
            return idx
        else:
            return (idx,)


INDEXER = Indexer()


class Slicer:
    """Helper for defining slices more easily which always returns slices"""

    def __getitem__(self, idx: slice) -> slice:
        assert isinstance(idx, slice)
        return idx


SLICER = Slicer()
