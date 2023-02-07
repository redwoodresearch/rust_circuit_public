import importlib.util
import uuid
from pathlib import Path
from typing import Optional

import pytest
import torch

import rust_circuit as rc
from interp.tools.rrfs import RRFS_DIR


def test_basic_funcs_correct():
    tester = rc.GeneralFunctionSpecTester(test_with_rand=True, start_shape_num=1)
    for name in [
        "sigmoid",
        "tanh",
        # "rsqrt",
        "gelu",
        "relu",
        "step",
        "reciprocal",
        "log_exp_p_1",
        "gaussian_pdf",
        "gaussian_cdf",
        "softmax",
        "log_softmax",
        # "q_from_qr",
        "min",
        "max",
    ]:
        tester.test_many_shapes(
            rc.GeneralFunction.new_by_name(rc.Symbol.new_with_random_uuid((3, 3, 3)), spec_name=name).spec
        )


def test_cast():
    arr = rc.Array.randn(1, 2, device_dtype=rc.TorchDeviceDtypeOp("cpu", "float16"))
    casty1 = rc.GeneralFunction.new_cast(
        arr, rc.TorchDeviceDtypeOp("cpu", torch.float16), rc.TorchDeviceDtypeOp("cpu", torch.float64)
    )
    assert casty1.device_dtype == rc.TorchDeviceDtypeOp("cpu", "float64")
    assert casty1.evaluate().dtype == torch.float64
    print(casty1, casty1.device_dtype, casty1.evaluate())

    with pytest.raises(rc.MiscInputCastIncompatibleDeviceDtypeError):
        casty2 = rc.GeneralFunction.new_cast(
            arr, rc.TorchDeviceDtypeOp("cpu", "int16"), rc.TorchDeviceDtypeOp("cpu", "float64")
        )
        print(casty2, casty2.device_dtype, casty2.evaluate())


@pytest.mark.cuda
def test_cast_cuda():
    arr = rc.Array.randn(1, 2, device_dtype=rc.TorchDeviceDtypeOp("cpu", "float16"))
    casty1 = rc.GeneralFunction.new_cast(
        arr, rc.TorchDeviceDtypeOp("cpu", "float16"), rc.TorchDeviceDtypeOp("cuda:0", "float64")
    )
    assert casty1.device_dtype == rc.TorchDeviceDtypeOp("cuda:0", "float64")
    assert casty1.evaluate().dtype == torch.float64


def test_custom_generalfunction_device_dtype():
    class MyCustomSigmoid(rc.GeneralFunctionSpecBase):
        @property
        def name(self) -> str:
            return "my_sigmoid"

        def compute_hash_bytes(self) -> bytes:
            return uuid.UUID("36a3ef34-a3cb-49a5-8cae-89f29f269d78").bytes

        def function(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.sigmoid(x)

        def get_shape_info(self, *shapes: rc.Shape) -> rc.GeneralFunctionShapeInfo:
            return rc.get_shape_info_simple(shapes)

        @classmethod
        def new(cls, circ: rc.Circuit, name: Optional[str] = None):
            """convenience function"""
            return rc.GeneralFunction(circ, spec=cls(), name=name)

        # optionally you can implement `get_device_dtype_override`, which validates input device/dtypes and optionally returns the output device/dtype
        def get_device_dtype_override(self, *device_dtypes: rc.TorchDeviceDtypeOp) -> Optional[rc.TorchDeviceDtypeOp]:
            print("DOING OVERRIDE")
            assert device_dtypes[0].dtype in {
                "float16",
                "float32",
                "float64",
                "bfloat16",
            }  # softmax only applies to floats, not ints
            print("overrided")
            return None  # don't change device/dtype

    with pytest.raises(AssertionError):
        rc.GeneralFunction(rc.Array(torch.randint(10, 30, (2, 3))), spec=MyCustomSigmoid())
    rc.GeneralFunction(rc.Array(torch.randn(10, 30)), spec=MyCustomSigmoid())


def test_new_by_path():
    path = Path(RRFS_DIR) / "test" / "general_function.py"
    spec = importlib.util.spec_from_file_location("user_funcs", str(path))
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    spec = foo.MyCustomSigmoid()
    x = rc.Symbol.new_with_random_uuid((3, 3, 3))
    f_by_path = rc.GeneralFunction.new_by_path(x, path="/test/general_function.py:MyCustomSigmoid")
    f_by_spec = rc.GeneralFunction(x, spec=spec)
    assert f_by_path == f_by_spec
    assert f_by_path == rc.Parser()(f_by_path.repr())
    assert f_by_spec == rc.Parser()(f_by_spec.repr())
