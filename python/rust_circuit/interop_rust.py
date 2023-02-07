import functools
import hashlib
import weakref
from typing import *

import torch

import rust_circuit.optional as op
from interp.circuit.circuit import Circuit, MemoizedFn, Shape
from interp.circuit.circuit_compiler.util import TorchAxisIndex
from interp.circuit.circuit_utils import device_eq, evaluate_fn
from interp.circuit.computational_node import Add, Concat, Einsum, GeneralFunction, Index, UnaryRearrange, WildFunction
from interp.circuit.constant import ArrayConstant, FloatConstant, One, Zero
from interp.circuit.cumulant import Cumulant
from interp.circuit.eq_by_big_hash import hash_add, hash_add_by_id
from interp.circuit.var import AutoTag, DiscreteVar, StoredCumulantVar
from rust_circuit.py_utils import assert_never

from . import _rust as rc
from ._rust import Add as rAdd
from ._rust import Array as rArray
from ._rust import Circuit as rCircuit
from ._rust import Concat as rConcat
from ._rust import Cumulant as rCumulant
from ._rust import DiscreteVar as rDiscreteVar
from ._rust import Einsum as rEinsum
from ._rust import GeneralFunction as rGeneralFunction
from ._rust import GeneralFunctionShapeInfo, GeneralFunctionSimpleSpec, GeneralFunctionSpecBase
from ._rust import Index as rIndex
from ._rust import Rearrange as rRearrange
from ._rust import Scalar as rScalar
from ._rust import Scatter as rScatter
from ._rust import Schedule as rSchedule
from ._rust import StoredCumulantVar as rStoredCumulantVar
from ._rust import Tag as rTag
from ._rust import TorchDeviceDtypeOp, scatter_to_concat

MYPY = False


class CantConvertCircuitError(Exception):
    def __init__(self):
        try:
            import hypothesis
        except:
            super().__init__()
            return
        if hypothesis.currently_in_test_context():
            hypothesis.assume(False)
        super().__init__()


class FromPyGeneralFunctionSpec(GeneralFunctionSpecBase):
    name_val: str
    function_val: Callable  # Callable[[*torch.Tensor], torch.Tensor]
    get_shape_val: Callable  # Callable[[*Shape], Optional[Sequence[int]]]
    num_non_batchable_output_dims: int
    input_batchability: Tuple[bool, ...]

    def __init__(
        self,
        name: str,
        function: Callable,
        get_shape: Callable,
        num_non_batchable_output_dims: int,
        input_batchability: Tuple[bool, ...],
    ) -> None:
        self.name_val = name
        if not MYPY:  # this works on newer versions of MYPY - likely a bug
            self.function_val = function
            self.get_shape_val = get_shape
        self.num_non_batchable_output_dims = num_non_batchable_output_dims
        self.input_batchability = input_batchability

    @property
    def name(self) -> str:
        return self.name_val

    def compute_hash_bytes(self) -> bytes:
        m = hashlib.blake2b(str(self.__class__).encode())
        hash_add(m, self.name)
        hash_add_by_id(m, self.function_val)
        hash_add_by_id(m, self.get_shape_val)
        hash_add(m, self.num_non_batchable_output_dims)
        hash_add(m, self.input_batchability)

        return m.digest()

    def function(self, *tensors: torch.Tensor) -> torch.Tensor:
        return self.function_val(*tensors)

    def get_shape_info(self, *shapes: Shape) -> GeneralFunctionShapeInfo:
        return GeneralFunctionShapeInfo(
            self.get_shape_val(*shapes), self.num_non_batchable_output_dims, self.input_batchability
        )


def py_to_rust(py: Circuit, device_dtype_op=TorchDeviceDtypeOp()) -> rCircuit:
    @functools.cache
    def recurse(py: Circuit) -> rCircuit:
        if isinstance(py, (Zero, One, FloatConstant)):
            return rScalar(py.value, py.shape, py.name)
        elif isinstance(py, ArrayConstant):
            tensor = py.value
            if tensor.shape != py.shape:
                tensor = torch.broadcast_to(py.value, py.shape)
            if (
                device_dtype_op.dtype is not None
                and (torch_dtype := getattr(torch, device_dtype_op.dtype)) != tensor.dtype
            ):
                tensor = tensor.to(dtype=torch_dtype)
            if not device_eq(tensor, device_dtype_op.device):
                tensor = tensor.to(device=device_dtype_op.device)
            return rArray(tensor, py.name)
        elif isinstance(py, Add):
            return rAdd(*[recurse(x) for x in py.to_unweighted().items.keys()], name=py.name)
        elif isinstance(py, Einsum):
            return rEinsum(*[(recurse(operand), ints) for operand, ints in py.args], out_axes=py.out_axes, name=py.name)
        elif isinstance(py, UnaryRearrange):
            spec = py.get_spec()
            rust_child = recurse(py.node)
            return rRearrange(rust_child, spec.to_rust(), py.name)
        elif isinstance(py, Index):
            index = py.index
            new_idx: List[TorchAxisIndex] = []
            # convert slices from python approach
            for i, s in zip(index, py.node.shape):
                if (not MYPY) and isinstance(i, slice):
                    fix = lambda item: op.map(item, lambda x: max(min(x, s), -s))
                    i = slice(fix(i.start), fix(i.stop))
                elif (not MYPY) and isinstance(i, torch.Tensor) and i.ndim == 0:
                    i = int(i.item())

                new_idx.append(i)
            index = tuple(new_idx)

            if device_dtype_op.device is not None:
                index = tuple(
                    [
                        x.to(device=device_dtype_op.device)
                        if isinstance(x, torch.Tensor) and not device_eq(x, device_dtype_op.device)
                        else x
                        for x in index
                    ]
                )
            return rIndex(recurse(py.node), index, py.name)
        elif isinstance(py, Concat):
            return rConcat(*[recurse(x) for x in py.circuits], axis=py.axis, name=py.name)
        elif isinstance(py, AutoTag):
            return rTag(recurse(py.node), py.uuid, py.name)
        elif isinstance(py, GeneralFunction):
            recursed = recurse(py.node)
            if py.rust_spec is not None:
                return rGeneralFunction(recursed, spec=py.rust_spec, name=py.name)
            name = py.function.__name__.removesuffix("_fn") if hasattr(py.function, "__name__") else "unk_py_fn"
            result = rGeneralFunction.new_by_name_op(recursed, spec_name=name, name=py.name)
            if result is not None:
                return result
            else:
                num_non_batchable = len(py.normalized_non_batch_dims())

                gspec = FromPyGeneralFunctionSpec(
                    name,
                    py.function,
                    py_generalfunction_get_shape(num_non_batchable),
                    num_non_batchable,
                    (py.allows_batching,),
                )
                return rGeneralFunction(recursed, spec=gspec, name=py.name)
        elif isinstance(py, WildFunction):
            # TODO: not well tested!!!
            rec_nodes = [recurse(n) for n in py.nodes]

            gspec = FromPyGeneralFunctionSpec(
                py.function.get_name()
                if hasattr(py.function, "get_name")
                else py.function.__name__.removesuffix("_fn"),
                py.function,
                py.get_wild_function_shape_getter(),
                py.num_non_batchable_output_dims,
                tuple(py.input_batchability),
            )
            return rGeneralFunction(*rec_nodes, spec=gspec, name=py.name)
        elif isinstance(py, Cumulant):
            return rCumulant(*[recurse(x) for x in py.circuits], name=py.name)
        elif isinstance(py, DiscreteVar):
            return rDiscreteVar(recurse(py.values), recurse(py.probs_and_group), py.name)
        elif isinstance(py, StoredCumulantVar):
            return rStoredCumulantVar.new_mv(
                recurse(py.mean),
                recurse(py.cov),
                {k: recurse(v) for k, v in py.higher_cumulants.items()},
                py.uuid,
                py.name,
            )
        else:
            raise NotImplementedError(f"py_to_rust for {py.__class__.__name__} unimplemented")

    return recurse(py)


def rust_to_py(rust: rCircuit):
    @functools.cache
    def recurse(rust: rCircuit):
        result = recurse_raw(rust)
        assert result.shape == rust.shape, (result.shape, rust.shape, rust, result)
        assert result.ndim == len(rust.shape)
        return result

    def recurse_raw(rust: rCircuit) -> Circuit:
        if isinstance(rust, rScalar):
            return FloatConstant(rust.value, rust.shape, rust.name)
        elif isinstance(rust, rArray):
            return ArrayConstant(rust.value, rust.value.shape, name=rust.name)
        elif isinstance(rust, rAdd):
            return Add.from_unweighted_list([recurse(x) for x in rust.children], rust.name)
        elif isinstance(rust, rEinsum):
            return Einsum.from_axes_tuples(
                *[(recurse(operand), ints) for operand, ints in rust.args], out_axes=rust.out_axes, name=rust.name
            )
        elif isinstance(rust, rRearrange):
            return UnaryRearrange.from_spec(
                recurse(rust.node),
                rust.spec.to_py_rearrange_spec(rust.node.shape),
                rust.name,
            )
        elif isinstance(rust, rIndex):
            return Index(recurse(rust.node), tuple(rust.idx), name=rust.name)
        elif isinstance(rust, rGeneralFunction):
            spec = rust.spec
            if not (
                isinstance(spec, (GeneralFunctionSimpleSpec, FromPyGeneralFunctionSpec))
                and (rust.children[0].shape == rust.shape)
                and (len(rust.children) == 1)
            ):
                raise CantConvertCircuitError()

            if isinstance(spec, GeneralFunctionSimpleSpec):
                allows_batching = True
                function = spec.get_function()
            elif isinstance(spec, FromPyGeneralFunctionSpec):
                allows_batching = spec.input_batchability[0]
                function = spec.function_val
            else:
                assert_never(spec)

            return GeneralFunction(
                recurse(rust.children[0]),
                function,
                None,
                name=rust.name,
                allows_batching=allows_batching,  # this is always batchable
                non_batch_dims=tuple(range(-spec.num_non_batchable_output_dims, 0)),
                rust_spec=spec,
            )
        elif isinstance(rust, rConcat):
            return Concat(tuple([recurse(x) for x in rust.children]), rust.axis, rust.name)
        elif isinstance(rust, rScatter):
            return recurse(scatter_to_concat(rust))
        elif isinstance(rust, rTag):
            return AutoTag(recurse(rust.node), rust.uuid, rust.name)
        elif isinstance(rust, rDiscreteVar):
            return DiscreteVar(recurse(rust.values), recurse(rust.probs_and_group), rust.name)
        elif isinstance(rust, rStoredCumulantVar):
            highers = {k: recurse(v) for k, v in rust.cumulants.items() if k != 1 and k != 2}
            return StoredCumulantVar(
                recurse(rust.cumulants[1]), recurse(rust.cumulants[2]), highers, rust.uuid, rust.name
            )
        elif isinstance(rust, rCumulant):
            return Cumulant(tuple([recurse(x) for x in rust.children]), rust.name)
        elif isinstance(rust, rc.Module):
            raise CantConvertCircuitError()
        else:
            raise NotImplementedError(rust)

    return recurse(rust)


def py_generalfunction_get_shape(num_non_batchable: int):
    def check(shape: Shape):
        assert len(shape) >= num_non_batchable
        return shape

    return check


# this takes python circuit, rust version takes rust circuit
def schedule_replace_circuits(schedule: rSchedule, map: Dict[Circuit, torch.Tensor]):
    new_dict = {py_to_rust(k).hash: v for k, v in map.items()}
    result = schedule.replace_tensors(new_dict)
    return result


def rust_get_f64_evaluator():
    return lambda circuits: [MemoizedFn(evaluate_fn(dtype=torch.float64))(c) for c in circuits]


def evaluate_py(pycirc):
    return py_to_rust(pycirc).evaluate()


def eval_opt_f64_py(pycirc):
    return rc.optimize_and_evaluate(
        py_to_rust(pycirc, TorchDeviceDtypeOp(dtype="float64")), rc.OptimizationSettings()
    ).to(dtype=torch.float64)


def cached_circuit_by_hash(fn):
    """Caches a function Circuit->Any, only storing hash of input"""
    cachy: Dict[Tuple, Any] = {}

    @functools.wraps(fn)
    def wrapped_fn(circuit: rCircuit, *other_args):
        nonlocal cachy
        key = tuple([circuit.hash, *other_args])
        if key in cachy:
            return cachy[key]
        result = fn(circuit, *other_args)
        cachy[key] = result
        return result

    return wrapped_fn


def cached_circuit_by_hash_weak(fn):
    """Caches a function Circuit->Any, only storing hash of input"""
    cachy: Dict[Tuple, weakref.ref[Any]] = {}

    @functools.wraps(fn)
    def wrapped_fn(circuit: rCircuit, *other_args):
        nonlocal cachy
        key = tuple([circuit.hash, *other_args])
        if key in cachy and (cached_result := cachy[key]()) is not None:
            return cached_result
        result = fn(circuit, *other_args)
        cachy[key] = weakref.ref(result)
        return result

    return wrapped_fn
