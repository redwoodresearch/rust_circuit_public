from __future__ import annotations

import abc
from typing import Optional, Tuple

import torch


class GeneralFunctionShapeInfo:
    ...


class TorchDeviceDtypeOp:
    ...


# this class is duplicated in rust_circuit.pyi due to unfortunate details of
# how python type stubs work
# so changes here should be replicated there etc
class GeneralFunctionSpecBase(metaclass=abc.ABCMeta):
    """
    Inherit from this base class in order to implement an arbitrary new GeneralFunctionSpec.

    See docs for `get_shape_info`, GeneralFunctionShapeInfo, and `function`.
    """

    @abc.abstractproperty
    def name(self) -> str:
        raise NotImplementedError

    @property
    def path(self) -> Optional[str]:
        """The path of this spec relative to RRFS_DIR if this is stored in rrfs.
        Should be of the form /dir/sub_dir/.../file.py:MyCustomSpec"""
        return None

    def compute_hash_bytes(self) -> bytes:
        """the default implementation should typically be overridden!"""
        return id(self).to_bytes(8, "big", signed=True)

    @abc.abstractmethod
    def function(self, *tensors: torch.Tensor) -> torch.Tensor:
        """run the actual function on tensors - these tensors shapes correspond to the shapes in ``get_shape_info``"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape_info(self, *shapes: Tuple[int, ...]) -> GeneralFunctionShapeInfo:
        """This should error (exception) if the shapes are invalid and otherwise return GeneralFunctionShapeInfo"""
        raise NotImplementedError

    def get_device_dtype_override(self, *device_dtypes: TorchDeviceDtypeOp) -> Optional[TorchDeviceDtypeOp]:
        """Use this to check the validity of input dtypes/devices and set output device/dtype. If this returns None, the output inherits device/dtype from children.
        This can error to signal that inputs have invalid dtypes/devices"""
        return None
