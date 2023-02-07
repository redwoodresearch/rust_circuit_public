import abc
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Sequence, Set, Tuple, Type, Union
from uuid import UUID

import torch

# note that functions which return 'Shape' should return PyShape in rust!
Shape = Tuple[int, ...]

class RearrangeSpec(object):
    input_ints: List[List[int]]
    output_ints: List[List[int]]
    int_sizes: List[Optional[int]]
    def __init__(
        self,
        input_ints: List[List[int]],
        output_ints: List[List[int]],
        int_sizes: List[Optional[int]],
    ): ...
    def is_identity(self) -> bool: ...
    def is_permute(self) -> bool: ...
    def shapes(self) -> Tuple[List[int], List[int]]: ...
    def is_valid(self) -> bool: ...
    def to_einops_string(self) -> str:
        """alias for self.to_string(False, False, False)"""
    def to_string(
        self, size_on_letter: bool = True, use_symbolic: bool = True, show_size_if_present: bool = True
    ) -> str: ...
    @staticmethod
    def from_string(string: str) -> RearrangeSpec: ...
    def apply(self, tensor: torch.Tensor) -> torch.Tensor: ...
    @staticmethod
    def fuse(inner: RearrangeSpec, outer: RearrangeSpec) -> RearrangeSpec: ...
    def canonicalize(self, special_case_ones: bool = True) -> RearrangeSpec: ...
    def fill_empty_ints(self) -> RearrangeSpec: ...
    def conform_to_input_shape(self, shape: Shape) -> RearrangeSpec: ...
    def conform_to_output_shape(self, shape: Shape) -> RearrangeSpec: ...
    def next_axis(self) -> int: ...
    def expand_to(self, shape: Shape) -> RearrangeSpec: ...
    @staticmethod
    def ident(rank: int) -> RearrangeSpec: ...
    def to_py_rearrange_spec(self, shape: Shape) -> Any: ...
    @staticmethod
    def flatten(rank: int) -> RearrangeSpec: ...
    @staticmethod
    def unflatten(shape: Sequence[int]) -> RearrangeSpec: ...
    @staticmethod
    def unflatten_axis(ndim: int, axis: int, shape: Sequence[int]) -> RearrangeSpec: ...
    @staticmethod
    def check_rank(rank: int) -> None: ...
    @staticmethod
    def expand_at_axes(
        orig_ndim: int, axes: Union[int, Sequence[int]], counts: Optional[Union[int, Sequence[int]]] = None
    ) -> RearrangeSpec:
        """
        if counts is None, this does unsqueeze (aka, uses counts = [1, ..., 1])
        """
    @staticmethod
    def unsqueeze(orig_ndim: int, axes: Union[int, Sequence[int]]) -> RearrangeSpec:
        """
        alias for expand_at_axes with counts=None
        """
    @staticmethod
    def prepend_batch_shape(shape: Sequence[int], old_rank: int) -> RearrangeSpec:
        """meant to replace many use cases of python `repeat_at_axes` (now 'expand_at_axes' in rust)"""
        ...

Axis = Union[None, int, Sequence[int]]

# TODO: below could be subclasses of circuit if we wanted. This could be nice in various ways...

IrreducibleNode = Union[Array, Symbol]
"""
These nodes are unaffected by rewrites, and satisfy
AlgebraicRewrite(Replace(X, IrreducibleNode->Y)) == Replace(AlgebraicRewrite(IrreducibleNode), IrreducibleNode->Y)
except for hashmap iteration order or other unfortunate nondeterminism
"""

Leaf = Union[Array, Symbol, Scalar]
LeafConstant = Union[Array, Scalar]
Var = Union[StoredCumulantVar, DiscreteVar]
Path = List[Circuit]  # First circuit is matched circuit, last circuit is the root
Paths = List[Path]

class Circuit:
    @property
    def shape(self) -> Shape: ...
    @property
    def is_constant(self) -> bool: ...
    @property
    def is_explicitly_computable(self) -> bool: ...
    @property
    def can_be_sampled(self) -> bool: ...
    @property
    def no_unbound_symbols(self) -> bool: ...
    @property
    def use_autoname(self) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def op_name(self) -> Optional[str]: ...
    def get_autoname(self) -> Optional[str]: ...
    @property
    def symbolic_size_constraints(self) -> Set[SymbolicSizeConstraint]: ...
    @property
    def hash(self) -> bytes: ...
    @property
    def hash_base16(self) -> str: ...
    @property
    def device_dtype(self) -> TorchDeviceDtypeOp: ...
    @property
    def device(self) -> Optional[str]: ...
    @property
    def dtype(self) -> Optional[str]: ...
    @property
    def torch_dtype(self) -> Optional[torch.dtype]: ...
    @property
    def num_children(self) -> int: ...
    @property
    def children(self) -> List[Circuit]: ...
    @property
    def non_free_children(self) -> List[Circuit]: ...
    @property
    def rank(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def get_compatible_device_dtype(self) -> TorchDeviceDtype: ...
    @property
    def numel(self) -> int: ...
    def self_flops(self) -> int: ...
    def total_flops(self) -> int: ...
    def max_non_input_size(self) -> int: ...
    def print(self, options: PrintOptionsBase = PrintOptions()) -> None: ...
    def string_escape(self, name: str) -> str: ...
    def repr(self, options: PrintOptions = PrintOptions()) -> str: ...
    def print_stats(self) -> None: ...
    def print_html(self, options: PrintHtmlOptions = PrintHtmlOptions()) -> None: ...
    def to_py(self) -> Any: ...
    def evaluate(self) -> torch.Tensor: ...
    def map_children(self, fn: Callable[[Circuit], Circuit]) -> Circuit: ...
    def map_children_enumerate(self, fn: Callable[[int, Circuit], Circuit]) -> Circuit: ...
    def total_arrayconstant_size(self) -> int: ...
    def get_device_dtype(self) -> TorchDeviceDtypeOp: ...
    def rename(self, name: Optional[str]) -> Circuit: ...
    def with_autoname_disabled(self, autoname_disabled: bool = True) -> Circuit: ...
    def rename_axes(self, named_axes: Dict[int, str]) -> Circuit:
        """Return set_named_axes(self, named_axes)."""
    def visit(self, f: Callable[[Circuit], None]): ...
    def update(
        self,
        matcher: IterativeMatcherIn,
        transform: TransformIn,
        cache_transform: bool = True,
        cache_update: bool = True,
        fancy_validate: bool = False,
        assert_any_found: bool = False,
    ) -> Circuit: ...
    def get(self, matcher: IterativeMatcherIn, fancy_validate: bool = False) -> Set[Circuit]:
        """Returns the set of all nodes within the circuit which match."""
    def get_unique_op(self, matcher: IterativeMatcherIn, fancy_validate: bool = False) -> Optional[Circuit]: ...
    def get_unique(self, matcher: IterativeMatcherIn, fancy_validate: bool = False) -> Circuit: ...
    def get_paths(self, matcher: IterativeMatcherIn) -> Dict[Circuit, Path]:
        """Return a dict where each matched node is associated with one arbitrary path to it."""
    def get_all_paths(self, matcher: IterativeMatcherIn) -> Dict[Circuit, Paths]:
        """Return a dict where each matched node is associated with a list of every path to it."""
    def get_all_circuits_in_paths(self, matcher: IterativeMatcherIn) -> List[Circuit]: ...
    def are_any_found(self, matcher: IterativeMatcherIn) -> bool:
        """Returns true if any node within the circuit matches."""
    def symbolic_shape(self) -> List[SymbolicSizeProduct]: ...
    def add(self, other: Circuit, name: Optional[str] = None) -> Add: ...
    def sub(self, other: Circuit, name: Optional[str] = None) -> Add: ...
    def mul(self, other: Circuit, name: Optional[str] = None) -> Einsum: ...
    def mul_scalar(self, scalar: float, name: Optional[str] = None, scalar_name: Optional[str] = None) -> Einsum: ...
    def reduce(self, op_name: str, axis: Axis = None, name: Optional[str] = None) -> Circuit: ...
    def sum(self, axis: Axis = None, name: Optional[str] = None) -> Einsum: ...
    def mean(self, axis: Axis = None, name: Optional[str] = None, scalar_name: Optional[str] = None) -> Einsum: ...
    def max(self, axis: Axis = None, name: Optional[str] = None) -> Circuit: ...
    def min(self, axis: Axis = None, name: Optional[str] = None) -> Circuit: ...
    def index(self, index: Sequence[TorchAxisIndex], name: Optional[str] = None) -> Index: ...
    def expand_at_axes(
        self,
        axes: Union[int, Sequence[int]],
        counts: Optional[Union[int, Sequence[int]]] = None,
        name: Optional[str] = None,
    ) -> Rearrange:
        """
        if counts is None, this does unsqueeze (aka, uses counts = [1, ..., 1])
        """
    def unsqueeze(self, axes: Union[int, Sequence[int]], name: Optional[str] = None) -> Rearrange:
        """
        adds 1 dims, also alias for expand_at_axes with counts=None
        """
    def squeeze(self, axes: Union[int, Sequence[int]], name: Optional[str] = None) -> Rearrange:
        """
        removes 1 dims at idxs, errors if dims at idx aren't 1
        """
    def flatten(self, name: Optional[str] = None) -> Rearrange: ...
    def unflatten(self, shape: Shape, name: Optional[str] = None) -> Rearrange: ...
    def unflatten_axis(self, axis: int, shape: Shape, name: Optional[str] = None) -> Rearrange: ...
    def rearrange(self, spec: RearrangeSpec, name: Optional[str] = None) -> Rearrange:
        """
        Alias for Rearrange(self, spec, name)
        """
    def rearrange_str(self, string: str, name: Optional[str] = None) -> Rearrange:
        """
        Alias for Rearrange.from_string(self, string, name)
        """
    def enforce_dtype_device(self, device_dtype: TorchDeviceDtypeOp, name: Optional[str] = None) -> GeneralFunction:
        """
        Alias for GeneralFunction.new_cast(self, input_required_compatibility=device_dtype, output=device_dtype, name=name)
        """
    def maybe_einsum(self) -> Optional[Einsum]: ...
    def maybe_array(self) -> Optional[Array]: ...
    def maybe_symbol(self) -> Optional[Symbol]: ...
    def maybe_scalar(self) -> Optional[Scalar]: ...
    def maybe_add(self) -> Optional[Add]: ...
    def maybe_rearrange(self) -> Optional[Rearrange]: ...
    def maybe_index(self) -> Optional[Index]: ...
    def maybe_general_function(self) -> Optional[GeneralFunction]: ...
    def maybe_concat(self) -> Optional[Concat]: ...
    def maybe_scatter(self) -> Optional[Scatter]: ...
    def maybe_module(self) -> Optional[Module]: ...
    def maybe_tag(self) -> Optional[Tag]: ...
    def maybe_discrete_var(self) -> Optional[DiscreteVar]: ...
    def maybe_stored_cumulant_var(self) -> Optional[StoredCumulantVar]: ...
    def maybe_cumulant(self) -> Optional[Cumulant]: ...
    def cast_einsum(self) -> Einsum: ...
    def cast_array(self) -> Array: ...
    def cast_symbol(self) -> Symbol: ...
    def cast_scalar(self) -> Scalar: ...
    def cast_add(self) -> Add: ...
    def cast_rearrange(self) -> Rearrange: ...
    def cast_index(self) -> Index: ...
    def cast_general_function(self) -> GeneralFunction: ...
    def cast_concat(self) -> Concat: ...
    def cast_scatter(self) -> Scatter: ...
    def cast_module(self) -> Module: ...
    def cast_tag(self) -> Tag: ...
    def cast_discrete_var(self) -> DiscreteVar: ...
    def cast_stored_cumulant_var(self) -> StoredCumulantVar: ...
    def cast_cumulant(self) -> Cumulant: ...
    def is_einsum(self) -> bool: ...
    def is_array(self) -> bool: ...
    def is_symbol(self) -> bool: ...
    def is_scalar(self) -> bool: ...
    def is_add(self) -> bool: ...
    def is_rearrange(self) -> bool: ...
    def is_index(self) -> bool: ...
    def is_general_function(self) -> bool: ...
    def is_concat(self) -> bool: ...
    def is_scatter(self) -> bool: ...
    def is_module(self) -> bool: ...
    def is_tag(self) -> bool: ...
    def is_discrete_var(self) -> bool: ...
    def is_stored_cumulant_var(self) -> bool: ...
    def is_cumulant(self) -> bool: ...
    def is_irreducible_node(self) -> bool: ...
    def is_leaf(self) -> bool: ...
    def is_leaf_constant(self) -> bool: ...
    def is_var(self) -> bool: ...
    def into_irreducible_node(self) -> Optional[IrreducibleNode]: ...
    def into_leaf(self) -> Optional[Leaf]: ...
    def into_leaf_constant(self) -> Optional[LeafConstant]: ...
    def into_var(self) -> Optional[Var]: ...

class Array(Circuit):
    """
    An array literal. Similar to torch.tensor or np.array. It's good to name your array.

    If you want your Array on a particular device, make sure the tensor you pass in as a `value` to `__init__` is on the
    appropriate device.
    """

    def __init__(self, value: torch.Tensor, name: Optional[str] = None) -> None: ...
    @property
    def value(self) -> torch.Tensor: ...
    @staticmethod
    def randn(
        *shape: int,
        name: Optional[str] = None,
        device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp(),
        seed: Optional[int] = None,
    ) -> "Array": ...
    @staticmethod
    def save_rrfs(force=False) -> str: ...  # string is base16 key. if force, it will be pushed to rrfs
    @staticmethod
    def from_hash(name: Optional[str], hash_base16: str) -> "Array": ...
    @staticmethod
    def from_hash_prefix(name: Optional[str], hash_base16: str, cache: Optional[TensorCacheRrfs] = None) -> "Array": ...
    def tensor_hash_base16(self) -> str: ...

class Symbol(Circuit):
    def __init__(self, shape: Shape, uuid: UUID, name: Optional[str] = None) -> None: ...
    @property
    def uuid(self) -> UUID: ...
    @staticmethod
    def new_with_random_uuid(shape: Shape, name: Optional[str] = None) -> "Symbol": ...
    @staticmethod
    def new_with_none_uuid(shape: Shape, name: Optional[str] = None) -> "Symbol": ...

class Scalar(Circuit):
    def __init__(self, value: float, shape: Shape = (), name: Optional[str] = None) -> None: ...
    @property
    def value(self) -> float: ...
    def is_zero(self) -> bool: ...
    def is_one(self) -> bool: ...
    def evolve_shape(self, shape: Shape) -> Scalar: ...

class Einsum(Circuit):
    def __init__(
        self,
        *args: Tuple[Circuit, Tuple[int, ...]],
        out_axes: Tuple[int, ...],
        name: Optional[str] = None,
    ) -> None: ...
    @property
    def args(self) -> List[Tuple[Circuit, Tuple[int, ...]]]: ...
    @property
    def out_axes(self) -> Tuple[int, ...]: ...
    def all_input_circuits(self) -> List[Circuit]:
        """Deprecated, use .children instead"""
    def all_input_axes(self) -> List[Tuple[int, ...]]: ...
    @staticmethod
    def from_einsum_string(string: str, *nodes: Circuit, name: Optional[str] = None) -> Einsum: ...
    @staticmethod
    def from_fancy_string(string: str, *nodes: Circuit, name: Optional[str] = None) -> Einsum: ...
    @staticmethod
    def from_nodes_ints(
        nodes: Sequence[Circuit],
        input_axes: Sequence[Sequence[int]],
        output_ints: Sequence[int],
        name: Optional[str] = None,
    ) -> Einsum:
        """Error checking alias for main constructor"""
    @staticmethod
    def new_diag(node: Circuit, ints: List[int], name: Optional[str] = None) -> Einsum: ...
    @staticmethod
    def new_trace(node: Circuit, ints: List[int], name: Optional[str] = None) -> Einsum: ...
    @staticmethod
    def scalar_mul(
        node: Circuit, scalar: float, name: Optional[str] = None, scalar_name: Optional[str] = None
    ) -> Einsum: ...
    @staticmethod
    def elementwise_broadcasted(*nodes: Circuit, name: Optional[str] = None) -> Einsum: ...
    @staticmethod
    def empty(name: Optional[str] = None) -> Einsum: ...
    @staticmethod
    def identity(node: Circuit, name: Optional[str] = None) -> Einsum: ...
    @staticmethod
    def new_outer_product(
        *nodes: Circuit, name: Optional[str] = None, out_axes_permutation: Optional[List[int]] = None
    ) -> Einsum: ...
    def evolve(
        self,
        args: Optional[Sequence[Tuple[Circuit, Tuple[int, ...]]]] = None,
        out_axes: Optional[Tuple[int, ...]] = None,
    ) -> Einsum: ...
    def reduced_axes(self) -> Set[int]: ...
    def next_axis(self) -> int: ...
    def normalize_ints(self) -> Einsum: ...

class Add(Circuit):
    def __init__(self, *args: Circuit, name: Optional[str] = None) -> None: ...
    @property
    def nodes(self) -> List[Circuit]:
        """Deprecated, use .children instead"""
    def has_broadcast(self) -> bool: ...
    def nodes_and_rank_differences(self) -> List[Tuple[Circuit, int]]: ...
    def to_counts(self) -> Dict[Circuit, int]: ...
    # @staticmethod
    # def from_weighted_list(
    #     nodes_and_weights: Sequence[Tuple[Circuit, float]], use_1_weights: bool = False, name: Optional[str] = None
    # ) -> Add: ...
    @staticmethod
    def from_weighted_nodes(
        *args: Tuple[Circuit, float], use_1_weights: bool = False, name: Optional[str] = None
    ) -> Add: ...
    @staticmethod
    def minus(positive: Circuit, negative: Circuit, name: Optional[str] = None) -> Add: ...

class Rearrange(Circuit):
    def __init__(self, node: Circuit, spec: RearrangeSpec, name: Optional[str] = None) -> None: ...
    @staticmethod
    def from_string(node: Circuit, string: str, name: Optional[str] = None) -> Rearrange: ...
    @property
    def node(self) -> Circuit: ...
    @property
    def spec(self) -> RearrangeSpec: ...
    @staticmethod
    def reshape(node: Circuit, shape: Shape) -> Rearrange: ...

TorchAxisIndex = Union[int, slice, torch.Tensor]

class Index(Circuit):
    def __init__(
        self,
        node: Circuit,
        index: Sequence[TorchAxisIndex],
        name: Optional[str] = None,
    ) -> None: ...
    @property
    def node(self) -> Circuit: ...
    @property
    def idx(self) -> List[TorchAxisIndex]: ...
    @staticmethod
    def new_synchronized_to_start(
        node: Circuit, index: Sequence[TorchAxisIndex], name: Optional[str] = None
    ) -> Index: ...
    def child_axis_map_including_slices(self) -> List[List[Optional[int]]]: ...
    def has_tensor_axes(self) -> bool:
        """this is equivalent to 'is view (not copy) of child'"""

class Scatter(Circuit):
    """
    NOTE: Scatter is not well-supported and should not be used directly.
    (Instead you can do broadcasting multiplies with tensors filled with zeroes and ones.)

    Scatter is equivalent to:
    result = torch.zeros(shape)
    result[index] = node.evaluate()
    although index is considered dimwise

    Probably Scatter should be called PadZero instead.

    Currently rewrites only work with slice indices; maybe other indices will be supported in the future.
    """

    def __init__(
        self,
        node: Circuit,
        index: Sequence[TorchAxisIndex],
        shape: Tuple[int, ...],
        name: Optional[str] = None,
    ) -> None: ...
    @property
    def node(self) -> Circuit: ...
    @property
    def idx(self) -> List[TorchAxisIndex]: ...
    def is_identity(self) -> bool: ...

ConvPaddingShorthand = Union[int, List[int], List[Tuple[int, int]]]

class Conv(Circuit):
    def __init__(
        self,
        input: Circuit,
        filter: Circuit,
        stride: List[int],
        padding: ConvPaddingShorthand = 0,
        name: Optional[str] = None,
    ) -> None: ...
    @property
    def input(self) -> Circuit: ...
    @property
    def filter(self) -> Circuit: ...
    @property
    def strides(self) -> List[int]: ...
    @property
    def padding(self) -> List[Tuple[int, int]]: ...
    def is_identity(self) -> bool: ...

class GeneralFunctionShapeInfo:
    """
    Returned by ``get_shape_info``, this contains the shape itself as well as batchability info.

    input_batchability is a mask indicating which inputs support batching. if none do, there is no batching.
    the number of non batchable dims in output, starting from end, is num_non_batchable_output_dims.

    (TODO: see better docs/explainers on batching overall for general function)
    """

    shape: Shape
    num_non_batchable_output_dims: int
    input_batchability: Sequence[bool]
    def __init__(
        self, shape: Shape, num_non_batchable_output_dims: int, input_batchability: Sequence[bool]
    ) -> None: ...

# this class is duplicated in circuit_base/py_general_function_spec.py due to
# unfortunate details of how python type stubs work
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
    def get_shape_info(self, *shapes: Shape) -> GeneralFunctionShapeInfo:
        """This should error (exception) if the shapes are invalid and otherwise return GeneralFunctionShapeInfo"""
        raise NotImplementedError
    def get_device_dtype_override(self, *device_dtypes: TorchDeviceDtypeOp) -> Optional[TorchDeviceDtypeOp]:
        """Use this to check the validity of input dtypes/devices and set output device/dtype. If this returns None, the output inherits device/dtype from children.
        This can error to signal that inputs have invalid dtypes/devices"""
        return None

class GeneralFunctionSimpleSpec:
    """typically initialized via `new_by_name` method on GeneralFunction"""

    @property
    def name(self) -> str: ...
    @property
    def num_non_batchable_output_dims(self) -> int: ...
    @property
    def removed_from_end(self) -> int: ...
    @property
    def is_official(self) -> bool: ...
    def get_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """This exists for casting back to old python types and shouldn't typically be used..."""

def get_shape_info_simple(
    shapes: Sequence[Shape], num_non_batchable_output_dims: int = 0, removed_from_end: int = 0
) -> GeneralFunctionShapeInfo:
    """
    Compute shape info how GeneralFunctionSimpleSpec does it:
    - Error if len(shapes) != 1
    - allow for removing dims from end
    - and for having some fixed number of non batchable output dims (see ``GeneralFunctionShapeInfo`` docs)
    TODO: explain better
    """

def get_shape_info_broadcast(shapes: Sequence[Shape], special_case_ones: bool = True) -> GeneralFunctionShapeInfo:
    """
    allow broadcasts and compute batching accordingly
    """

class GeneralFunctionIndexSpec:
    """typically initialized via `gen_index` method on GeneralFunction"""

    @property
    def index_dim(self) -> int: ...
    @property
    def batch_x(self) -> bool: ...
    @property
    def batch_index(self) -> bool: ...
    @property
    def check_index_ints(self) -> bool: ...

class GeneralFunctionExplicitIndexSpec:
    """typically initialized via `explicit_index` method on GeneralFunction"""

    @property
    def index_dim(self) -> int: ...
    @property
    def x_non_batch_dims(self) -> int: ...
    @property
    def check_index_ints(self) -> bool: ...

class GeneralFunctionSetDDSpec:
    @property
    def input_required_compatibility(self) -> TorchDeviceDtypeOp: ...
    @property
    def output(self) -> TorchDeviceDtypeOp: ...

class GeneralFunctionPowSpec: ...

class GeneralFunctionMultinomialSpec:
    @property
    def replacement(self) -> bool: ...
    @property
    def shape(self) -> Shape: ...

GeneralFunctionSpec = Union[
    GeneralFunctionSpecBase,
    GeneralFunctionSimpleSpec,
    GeneralFunctionIndexSpec,
    GeneralFunctionExplicitIndexSpec,
    GeneralFunctionSetDDSpec,
    GeneralFunctionPowSpec,
    GeneralFunctionMultinomialSpec,
]
"""
GeneralFunctionSpec contains all needed info about function, and is the same on all instances with the same function.
"""

class GeneralFunction(Circuit):
    def __init__(
        self,
        *args: Circuit,
        spec: GeneralFunctionSpec,
        name: Optional[str] = None,
    ) -> None: ...
    @property
    def nodes(self) -> List[Circuit]:
        """Deprecated, use .children instead"""
    @property
    def spec(self) -> GeneralFunctionSpec: ...
    @staticmethod
    def new_from_parse(*nodes: Circuit, parse_string: str, name: Optional[str] = None) -> GeneralFunction: ...
    @staticmethod
    def new_by_name(*nodes: Circuit, spec_name: str, name: Optional[str] = None) -> GeneralFunction: ...
    @staticmethod
    def new_by_name_op(*nodes: Circuit, spec_name: str, name: Optional[str] = None) -> Optional[GeneralFunction]: ...
    @staticmethod
    def new_by_path(*nodes: Circuit, path: str, name: Optional[str] = None) -> GeneralFunction:
        """Load a spec from a path to a spec stored in rrfs.

        The path format should be /dir/sub_dir/.../file.py:MyCustomSpec
        where the path is relative to rrfs
        and where MyCustomSpec is a callable that returns an instance of GeneralFunctionSpecBase
        (you probably wants it to be the MyCustomSpec class itself).
        """
    @staticmethod
    def gen_index(
        x: Circuit,
        index: Circuit,
        index_dim: int,
        batch_x: bool = False,
        batch_index: bool = True,
        check_index_ints: bool = True,
        name: Optional[str] = None,
    ) -> GeneralFunction:
        """
        Be warned: shape computation depends on batch_x and batch_index!

        If both true, index shape is batch shape and this batch shape is
        aligned with x. (Also, index_dim refers to non-batch component!)

        If just batch_index, index_shape is batch shape.

        If just batch_x, x prior to index dim is batch shape (and index_dim
        must be negative). So shape is batch (from x), then index shape, then
        rest of x shape.

        If no batching, then shape computation is same as batch_x (but allows
        for positive index_dim).

        TODO: improve docs by showing shape examples.
        """
    @staticmethod
    def explicit_index(
        x: Circuit,
        index: Circuit,
        index_dim: int,
        x_non_batch_dims: int,
        check_index_ints: bool = True,
        name: Optional[str] = None,
    ) -> GeneralFunction:
        """
        similar to gen_index, but with different specification

        In particular, you say how many non_batch dims x has. Then, we can
        determine batching precisely based on this. (while supporting batching
        over both idx and x)
        """
    @staticmethod
    def new_cast(
        x: Circuit,
        input_required_compatibility: TorchDeviceDtypeOp = TorchDeviceDtypeOp(),
        output: TorchDeviceDtypeOp = TorchDeviceDtypeOp(),
        name: Optional[str] = None,
    ) -> GeneralFunction: ...

# any of the below function which operate on a particular dim always operate on
# last dim (e.g, softmax is on dim=-1)
# to generate below fns, `cargo run -p circuit_base --bin print_functions`
def sin(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def cos(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def sigmoid(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def tanh(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def rsqrt(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def gelu(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def gelu_new(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction:
    "name from huggingface transformers, not actually newer than gelu"

def relu(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def step(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def reciprocal(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def log_exp_p_1(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def gaussian_pdf(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def gaussian_cdf(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def softmax(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def log_softmax(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def min(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def max(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def last_dim_size(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def abs(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def exp(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def log(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def logit(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def pow(base: Circuit, exponent: Circuit, name: Optional[str] = None) -> GeneralFunction: ...
def multinomial(
    probs: Circuit, seed: Circuit, shape: Shape, replacement: bool = True, name: Optional[str] = None
) -> GeneralFunction: ...

class GeneralFunctionSpecTester:
    """
    min_frac_successful controls the minimum fraction of proposed shapes which don't error for 'test_many_shapes'.
    Specifically, we check if this fraction is sufficiently high for *any* of the different number of possible inputs shapes.

    min_frac_checked_batch is the same except for checking that batching works
    correctly (this fraction can be lower if some inputs shapes aren't
    batchable). If the function is never batchable, this should be set to 0.
    """

    samples_per_batch_dims: int
    base_shapes_samples: int
    min_frac_successful: float
    min_frac_checked_batch: float
    start_num_inputs: int
    end_num_inputs: int
    start_ndim: int
    end_ndim: int
    start_shape_num: int
    end_shape_num: int
    test_with_rand: bool
    randn_size_cap: int
    def __init__(
        self,
        samples_per_batch_dims: int = 3,
        base_shapes_samples: int = 100,
        min_frac_successful: float = 0.1,
        min_frac_checked_batch: float = 0.1,
        start_num_inputs: int = 0,
        end_num_inputs: int = 5,
        start_ndim: int = 0,
        end_ndim: int = 10,
        start_shape_num: int = 0,
        end_shape_num: int = 10,
        test_with_rand: bool = False,
        randn_size_cap: int = 1024 * 16,
    ) -> None: ...
    def test_from_shapes(
        self, spec: GeneralFunctionSpec, shapes: Sequence[Shape], shapes_must_be_valid: bool = False
    ): ...
    def test_many_shapes(self, spec: GeneralFunctionSpec): ...

class Concat(Circuit):
    def __init__(self, *args: Circuit, axis: int, name: Optional[str] = None) -> None: ...
    @property
    def nodes(self) -> List[Circuit]:
        """Deprecated, use .children instead"""
    @property
    def axis(self) -> int: ...
    def get_sizes_at_axis(self) -> List[int]: ...
    @staticmethod
    def stack(*args: Circuit, axis: int, name: Optional[str] = None) -> Concat: ...

class Module(Circuit):
    """
    Expects a spec that indicates what arguments can be replaced and **kwargs which indicates what values to replace
    them with (identified by name)
    """

    def __init__(self, spec: ModuleSpec, name: Optional[str] = None, **kwargs: Circuit) -> None: ...
    @staticmethod
    def new_flat(spec: ModuleSpec, *nodes: Circuit, name: Optional[str] = None) -> "Module":
        """Creates a new Module but uses the order of the nodes rather than their names to match the ModuleSpec"""
    @property
    def spec(self) -> ModuleSpec: ...
    @property
    def nodes(self) -> List[Circuit]: ...
    def substitute(
        self,
        name_prefix: Optional[str] = None,
        name_suffix: Optional[str] = None,
        use_self_name_as_prefix: bool = False,
    ) -> Circuit:
        """
        Returns the spec circuit, but replacing symbols with their bound values. This is an algebraic rewrite that eliminates this module.
        Also known as beta_reduce in lambda calculus: https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B2-reduction
        """
    def rename_on_replaced_path(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        use_self_name_as_prefix: bool = False,
    ) -> Module: ...
    def map_on_replaced_path(self, f: Callable[[Circuit], Circuit]) -> Module: ...
    def aligned_batch_shape(self) -> Shape: ...
    def conform_to_input_batch_shape(self, dims_to_conform: Optional[int] = None) -> Module:
        """None is conform all batch dims"""
    def conform_to_input_shapes(self) -> Module: ...
    def child_axis_map_inputs(self) -> List[List[Optional[int]]]: ...

def get_free_symbols(circuit: Circuit) -> List[Symbol]: ...
def clear_module_circuit_caches() -> None:
    """
    By default, we'll cache circuits resulting from various computations done
    for modules. This results in failing to free tensors etc. This function is
    a hack to solve this issue: it just clear the cache.
    """

class ModuleSpec(object):
    """This class maintains no invariants, but you can optionally check that args are present and arg names are unique"""

    circuit: Circuit
    arg_specs: List[ModuleArgSpec]
    def __init__(
        self,
        circuit: Circuit,
        arg_specs: Sequence[ModuleArgSpec],
        check_all_inputs_used: bool = True,
        check_unique_arg_names: bool = True,
    ) -> None: ...
    def check_all_inputs_used(self) -> None: ...
    def check_unique_arg_names(self) -> None: ...
    def map_circuit(self, f: Callable[[Circuit], Circuit]) -> ModuleSpec: ...
    @staticmethod
    def new_free_symbols(circuit: Circuit, check_unique_arg_names: bool = True) -> ModuleSpec: ...
    @staticmethod
    def new_extract(
        circuit: Circuit,
        arg_specs: Sequence[Tuple[Circuit, ModuleArgSpec]],
        check_all_inputs_used: bool = True,
        check_unique_arg_names: bool = True,
    ) -> ModuleSpec: ...
    def resize(self, shapes: Sequence[Shape]) -> ModuleSpec: ...
    def are_args_used(self) -> List[bool]: ...
    def map_on_replaced_path(self, f: Callable[[Circuit], Circuit]) -> ModuleSpec: ...
    def rename_on_replaced_path(
        self, name_prefix: Optional[str] = None, name_suffix: Optional[str] = None
    ) -> ModuleSpec: ...

class ModuleArgSpec(object):
    """
    batchable: can this input have extra batch dimensions added?
    expandable: can the non-batch dimensions have their sizes edited?
    ban_non_symbolic_size_expand: should expanding require that the symbol's size was symbolic at that dimension?
    """

    symbol: Symbol
    batchable: bool
    expandable: bool
    ban_non_symbolic_size_expand: bool
    def __init__(
        self, symbol: Symbol, batchable: bool = True, expandable: bool = True, ban_non_symbolic_size_expand: bool = True
    ) -> None: ...
    @staticmethod
    def just_name_shape(
        circuit: Circuit, batchable: bool = True, expandable: bool = True, ban_non_symbolic_size_expand: bool = False
    ) -> ModuleArgSpec: ...

# TODO: add more methods to these 2 as needed (mostly just here for error conversion)
class SymbolicSizeProduct:
    other_factor: int
    symbolic_sizes: List[int]

class SymbolicSizeConstraint:
    l: SymbolicSizeProduct
    r: SymbolicSizeProduct

class SetSymbolicShape(Circuit):
    def __init__(self, node: Circuit, shape: Shape, name: Optional[str] = None) -> None: ...
    @staticmethod
    def some_set_neq(node: Circuit, shape: Shape, name: Optional[str] = None) -> SetSymbolicShape: ...
    @staticmethod
    def some_set_and_symbolic_neq(node: Circuit, shape: Shape, name: Optional[str] = None) -> SetSymbolicShape: ...

# variables

class Tag(Circuit):
    def __init__(self, node: Circuit, uuid: UUID, name: Optional[str] = None) -> None: ...
    @property
    def node(self) -> Circuit: ...
    @property
    def uuid(self) -> UUID: ...
    @staticmethod
    def new_with_random_uuid(node: Circuit, name: Optional[str] = None) -> Tag: ...

class DiscreteVar(Circuit):
    """
    Represents a probability distribution over values. These cannot be evaluated (and, therefore, any circuits that
    contain DiscreteVars cannot be evaluated). To evaluate the model, you must sample from the distribution using a
    Sampler.

    For example, to run with all inputs:

    >>> Sampler(RunDiscreteVarAllSpec([var.probs_and_group])).sample(circuit).evaluate()
    """

    def __init__(self, values: Circuit, probs_and_group: Optional[Circuit] = None, name: Optional[str] = None) -> None:
        """
        None probs_and_group create a new uniform group
        """
    @property
    def values(self) -> Circuit: ...
    @property
    def probs_and_group(self) -> Circuit: ...
    @staticmethod
    def new_uniform(values: Circuit, name: Optional[str] = None) -> DiscreteVar: ...
    @staticmethod
    def uniform_probs_and_group(size: int, name: Optional[str] = None) -> Tag: ...

class StoredCumulantVar(Circuit):
    def __init__(
        self, cumulants: Dict[int, Circuit], uuid: Optional[UUID] = None, name: Optional[str] = None
    ) -> None: ...
    @property
    def cumulants(self) -> Dict[int, Circuit]: ...
    @property
    def uuid(self) -> UUID: ...
    @staticmethod
    def new_mv(
        mean: Circuit,
        variance: Circuit,
        higher_cumulants: Dict[int, Circuit] = {},
        uuid: Optional[UUID] = None,
        name: Optional[str] = None,
    ) -> StoredCumulantVar: ...

class Cumulant(Circuit):
    def __init__(self, *args: Circuit, name: Optional[str] = None) -> None: ...
    @property
    def nodes(self) -> List[Circuit]:
        """Deprecated, use .children instead"""
    def equivalent_explicitly_computable_circuit(self) -> Circuit: ...
    @property
    def cumulant_number(self) -> int: ...
    @staticmethod
    def new_canon_rearrange(*args: Circuit, name: Optional[str] = None) -> Cumulant: ...

def broadcast_shapes(*shapes: Shape) -> Shape: ...

class TorchDeviceDtype(object):
    def __init__(self, device: str, dtype: Union[str, torch.dtype]) -> None: ...
    @property
    def device(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    def get_torch_dtype(self) -> torch.dtype: ...
    def op(self) -> TorchDeviceDtypeOp: ...
    def cast_tensor(self, tensor: torch.Tensor) -> torch.Tensor: ...
    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> TorchDeviceDtype: ...

class TorchDeviceDtypeOp(object):
    def __init__(self, device: Optional[str] = None, dtype: Optional[Union[str, torch.dtype]] = None) -> None: ...
    @staticmethod
    def default() -> TorchDeviceDtypeOp: ...
    @property
    def device(self) -> Optional[str]: ...
    @property
    def dtype(self) -> Optional[str]: ...
    def get_torch_dtype(self) -> Optional[torch.dtype]: ...

class NotSet: ...

NOT_SET = NotSet()

CliColor = Union[
    Literal["Green"],
    Literal["Red"],
    Literal["Blue"],
    Literal["Yellow"],
    Literal["Cyan"],
    Literal["Magenta"],
    Literal["White"],
    None,
    int,
]

CircuitColorer = Callable[[Circuit], CliColor]
CircuitCommenter = Callable[[Circuit], str]

class PrintOptions(object):
    """
    `traversal` will stop the print when the iterative matcher terminates, not when it finds something. It may be helpful to call `.filter(term_if_matches=True)` to have the matcher (and thus printing) terminate when it finds a match.
    (See other cases where traversals are used, TODO: central example/docs)
    """

    bijection: bool
    shape_only_when_necessary: bool
    leaves_on_top: bool
    arrows: bool
    force_use_serial_numbers: bool
    number_leaves: bool
    colorer: Optional[CircuitColorer]
    comment_arg_names: bool
    commenters: List[CircuitCommenter]
    sync_tensors: bool
    seen_children_same_line: bool
    only_child_below: bool
    tensor_index_literal: bool
    show_all_named_axes: bool
    skip_module_symbols: bool
    tensor_save_dir: Optional[str]  # defaults to dir set by env vars
    @property
    def traversal(self) -> IterativeMatcher: ...
    @traversal.setter
    def traversal(self, traversal: Optional[IterativeMatcherIn]) -> None: ...
    @property
    def reference_circuits(self) -> Dict[Circuit, str]: ...
    def __init__(
        self,
        bijection: bool = True,
        reference_circuits: Dict[Circuit, str] = {},
        reference_circuits_by_name: List[Circuit] = [],
        shape_only_when_necessary: bool = True,
        traversal: Optional[IterativeMatcherIn] = None,
        leaves_on_top: bool = False,
        arrows: bool = False,
        force_use_serial_numbers: bool = False,
        number_leaves: bool = False,
        colorer: Optional[Callable[[Circuit], CliColor]] = None,
        comment_arg_names: bool = False,
        commenters: List[CircuitCommenter] = [],
        sync_tensors: bool = False,
        seen_children_same_line: bool = False,
        only_child_below: bool = False,
        tensor_index_literal: bool = False,
        show_all_named_axes: bool = False,
        skip_module_symbols: bool = False,
        tensor_save_dir: Optional[str] = None,
    ) -> None: ...
    def evolve(
        self,
        bijection: Union[bool, NotSet] = NOT_SET,
        reference_circuits: Union[Dict[Circuit, str], NotSet] = NOT_SET,
        shape_only_when_necessary: Union[bool, NotSet] = NOT_SET,
        traversal: Union[Optional[IterativeMatcherIn], NotSet] = NOT_SET,
        leaves_on_top: Union[bool, NotSet] = NOT_SET,
        arrows: Union[bool, NotSet] = NOT_SET,
        force_use_serial_numbers: Union[bool, NotSet] = NOT_SET,
        number_leaves: Union[bool, NotSet] = NOT_SET,
        colorer: Union[Optional[Callable[[Circuit], CliColor]], NotSet] = NOT_SET,
        comment_arg_names: Union[bool, NotSet] = NOT_SET,
        commenters: Union[List[CircuitCommenter], NotSet] = NOT_SET,
        sync_tensors: Union[bool, NotSet] = NOT_SET,
        seen_children_same_line: Union[bool, NotSet] = NOT_SET,
        only_child_below: Union[bool, NotSet] = NOT_SET,
        tensor_index_literal: Union[bool, NotSet] = NOT_SET,
        show_all_named_axes: Union[bool, NotSet] = NOT_SET,
        skip_module_symbols: Union[bool, NotSet] = NOT_SET,
        tensor_save_dir: Union[Optional[str], NotSet] = NOT_SET,
    ) -> PrintOptions: ...
    @staticmethod
    def debug_default() -> PrintOptions: ...
    @staticmethod
    def type_nest_default(
        circuit_type: Type[Circuit], traversal: Optional[IterativeMatcherIn] = None
    ) -> PrintOptions: ...
    @staticmethod
    def einsum_nest_default(traversal: Optional[IterativeMatcherIn] = None) -> PrintOptions: ...
    @staticmethod
    def add_nest_default(traversal: Optional[IterativeMatcherIn] = None) -> PrintOptions: ...
    @property
    def is_print_as_tree(self) -> bool: ...
    def print(self, *circuits: Circuit) -> None: ...
    def repr(self, *circuits: Circuit) -> str: ...
    @staticmethod
    def repr_depth(*circuits: Circuit, end_depth: int) -> str: ...
    @staticmethod
    def print_depth(*circuits: Circuit, end_depth: int): ...
    @staticmethod
    def size_colorer() -> CircuitColorer: ...
    @staticmethod
    def hash_colorer() -> CircuitColorer: ...
    @staticmethod
    def type_colorer() -> CircuitColorer: ...
    @staticmethod
    def computability_colorer() -> CircuitColorer: ...
    @staticmethod
    def fixed_color(color: Optional[int]) -> CircuitColorer: ...
    @staticmethod
    def circuit_set_colorer(map: Dict[str, Circuit]) -> CircuitColorer:
        """color string is one of Red, Green, Blue, Cyan, Magenta, Yellow, White"""
    @staticmethod
    def size_threshold_commenter(threshold: int = 400_000_000) -> CircuitCommenter: ...
    @staticmethod
    def dtype_commenter(only_arrays: bool = False) -> CircuitCommenter: ...
    def repr_line_info(self, circ: Circuit) -> str: ...
    def repr_extra_info(self, circ: Circuit) -> str: ...

PrintOptionsBase = Union[PrintOptions, PrintHtmlOptions]

def set_debug_print_options(options: PrintOptions):
    """Globally set options used in rust error messages as well as python __repr__
    By default, reads RR_DEBUG_END_DEPTH uses that to set end depth on traversal leaving other options as default
    (RR_DEBUG_END_DEPTH can be set to 'None' or to a positive integer)
    """

CircuitHtmlColorer = Callable[[Circuit], Optional[str]]

class PrintHtmlOptions(object):
    """
    traversal will stop the print when the iterative matcher terminates, not when it finds something.
    You will be able to expand nodes which have been visited by overall_traversal.
    """

    shape_only_when_necessary: bool
    # Which nodes to display from the start
    @property
    def traversal(self) -> IterativeMatcher: ...
    @traversal.setter
    def traversal(self, traversal: IterativeMatcherIn) -> None: ...
    display_copy_button: bool
    colorer: CircuitHtmlColorer  # Should return a css color. If None, use the primary color
    primary_color: str
    number_leaves: bool
    keep_all_cells: bool  # Keep the data of all notebook cells alive? Change to Fals if the circuit is too big.
    comment_arg_names: bool
    commenters: List[CircuitCommenter]
    # Which nodes to load. Choose smaller depth if the circuit is too big.
    @property
    def overall_traversal(self) -> IterativeMatcher: ...
    @overall_traversal.setter
    def overall_traversal(self, traversal: IterativeMatcherIn) -> None: ...
    def __init__(
        self,
        shape_only_when_necessary: bool = True,
        traversal: IterativeMatcherIn = traverse_until_depth(3),
        display_copy_button: bool = True,
        colorer: CircuitHtmlColorer = PrintHtmlOptions.type_colorer(),
        primary_color: str = "lightgray",
        number_leaves: bool = False,
        keep_all_cells: bool = True,
        comment_arg_names: bool = False,
        commenters: List[CircuitCommenter] = [],
        overall_traversal: IterativeMatcherIn = IterativeMatcher(False),
    ) -> None: ...
    # Note: default values are sometimes lies, actual defaults are None, and the code replace them with
    # objects which have the same behavior as the one written in the sub file
    def evolve(
        self,
        shape_only_when_necessary: Union[bool, NotSet] = NOT_SET,
        traversal: Union[IterativeMatcherIn, NotSet] = NOT_SET,
        display_copy_button: Union[bool, NotSet] = NOT_SET,
        colorer: Union[Optional[Callable[[Circuit], str]], NotSet] = NOT_SET,
        primary_color: Union[str, NotSet] = NOT_SET,
        number_leaves: Union[bool, NotSet] = NOT_SET,
        keep_all_cells: Union[bool, NotSet] = NOT_SET,
        comment_arg_names: Union[bool, NotSet] = NOT_SET,
        commenters: Union[List[CircuitCommenter], NotSet] = NOT_SET,
        overall_traversal: Union[IterativeMatcherIn, NotSet] = NOT_SET,
    ) -> PrintHtmlOptions: ...
    def light(self) -> PrintHtmlOptions: ...  # switch to light mode
    def print(self, *circuits: Circuit) -> None: ...
    def repr(self, *circuits: Circuit) -> str: ...
    @staticmethod
    def print_depth(*circuits: Circuit, depth: int = 3): ...
    @staticmethod
    def type_nest_default(
        circuit_type: Type[Circuit], traversal: Optional[IterativeMatcherIn] = None
    ) -> PrintHtmlOptions: ...
    @staticmethod
    def einsum_nest_default(traversal: Optional[IterativeMatcherIn] = None) -> PrintHtmlOptions: ...
    @staticmethod
    def add_nest_default(traversal: Optional[IterativeMatcherIn] = None) -> PrintHtmlOptions: ...
    # Colorers made for being used in dark mode
    @staticmethod
    def size_colorer() -> CircuitHtmlColorer: ...
    @staticmethod
    def hash_colorer() -> CircuitHtmlColorer: ...
    @staticmethod
    def type_colorer() -> CircuitHtmlColorer: ...
    @staticmethod
    def computability_colorer() -> CircuitHtmlColorer: ...
    @staticmethod
    def fixed_color(color: str) -> CircuitHtmlColorer: ...

class Parser:
    """
    Can deserialize a circuit from a string serialized with circuit.repr(). It loads tensors based on their hashes.
    """

    reference_circuits: Dict[str, Circuit]
    tensors_as_random: bool
    tensors_as_random_device_dtype: TorchDeviceDtypeOp
    allow_hash_with_random: bool
    on_repeat_check_info_same: bool
    module_check_all_inputs_used: bool
    module_check_unique_arg_names: bool
    def __init__(
        self,
        reference_circuits: Dict[str, Circuit] = {},
        reference_circuits_by_name: List[Circuit] = [],
        tensors_as_random: bool = False,
        tensors_as_random_device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp(),
        allow_hash_with_random: bool = False,
        on_repeat_check_info_same: bool = True,
        module_check_all_inputs_used: bool = True,
        module_check_unique_arg_names: bool = False,
    ) -> None: ...
    def __call__(self, string: str, tensor_cache: Optional[TensorCacheRrfs] = None) -> Circuit: ...
    def parse_circuit(self, string: str, tensor_cache: Optional[TensorCacheRrfs] = None) -> Circuit: ...
    def parse_circuits(self, string: str, tensor_cache: Optional[TensorCacheRrfs] = None) -> List[Circuit]: ...

class SimpFnSubset:
    @staticmethod
    def all_names() -> List[str]: ...
    @property
    def values(self) -> Dict[str, bool]: ...
    @staticmethod
    def init_all_with(include: bool) -> SimpFnSubset:
        """include/exclude every func (set all to ``include``)"""
    @staticmethod
    def none() -> SimpFnSubset:
        """exclude every func (set all to False)"""
    @staticmethod
    def all() -> SimpFnSubset:
        """include every func (set all to True)"""
    def none_repr(self) -> str:
        """like default repr, but use SimpFnSubset.none() instead of SimpFnSubset.default() as the base"""
    def check_names(self, names: Sequence[str]) -> None: ...
    def set_all_to(self, names: Sequence[str], include: bool) -> SimpFnSubset: ...
    def exclude(self, names: Sequence[str]) -> SimpFnSubset:
        """
        Alias for self.set_all_to(names, False)
        """
    def include(self, names: Sequence[str]) -> SimpFnSubset:
        """
        Alias for self.set_all_to(names, True)
        """
    def simp_step(self, circuit: Circuit) -> Optional[Circuit]: ...
    def simp(self, circuit: Circuit) -> Circuit: ...
    # to generate below 3 fns, `cargo run -p circuit_rewrites --bin print_simp_fn_stub_part`
    @staticmethod
    def compiler_default() -> SimpFnSubset:
        """
        Get compiler default simp fns. This is::
            SimpFnSubset.none().set(
                # Einsum
                einsum_elim_zero = True,
                einsum_elim_identity = True,
                einsum_flatten = True,
                einsum_of_permute_merge = True,
                einsum_merge_scalars = True,
                einsum_remove_one = False,
                einsum_pull_removable_axes = True,
                einsum_elim_removable_axes_weak = False,
                einsum_permute_to_rearrange = True,
                einsum_pull_scatter = True,
                einsum_push_down_trace = True,
                einsum_concat_to_add = True,
                # Array (none)
                # Symbol (none)
                # Scalar (none)
                # Add
                remove_add_few_input = True,
                add_flatten = True,
                add_elim_zeros = True,
                add_collapse_scalar_inputs = True,
                add_deduplicate = True,
                add_pull_removable_axes = True,
                add_pull_scatter = True,
                add_pull_diags = True,
                add_fuse_scalar_multiples = True,
                add_elim_removable_axes_weak = False,
                # Rearrange
                rearrange_elim_identity = True,
                rearrange_fuse = True,
                rearrange_merge_scalar = True,
                permute_of_einsum_merge = True,
                # Index
                index_elim_identity = True,
                index_fuse = True,
                index_merge_scalar = True,
                index_einsum_to_scatter = True,
                index_concat_drop_unreached = True,
                # GeneralFunction
                generalfunction_pull_removable_axes = True,
                generalfunction_merge_inverses = True,
                generalfunction_special_case_simplification = True,
                generalfunction_evaluate_simple = True,
                # Concat
                concat_elim_identity = True,
                concat_elim_split = True,
                concat_pull_removable_axes = True,
                concat_merge_uniform = True,
                concat_drop_size_zero = True,
                concat_fuse = True,
                concat_repeat_to_rearrange = True,
                concat_to_scatter = True,
                # Scatter
                scatter_elim_identity = True,
                scatter_fuse = True,
                scatter_pull_removable_axes = True,
                # Conv (none)
                # Module
                elim_empty_module = True,
                elim_no_input_module = True,
                # SetSymbolicShape (none)
                # Tag (none)
                # DiscreteVar (none)
                # StoredCumulantVar (none)
                # Cumulant (none)
            )
        """
    @staticmethod
    def default() -> SimpFnSubset:
        """
        Get default simp fns. This is::
            SimpFnSubset.none().set(
                # Einsum
                einsum_elim_zero = True,
                einsum_elim_identity = True,
                einsum_flatten = False,
                einsum_of_permute_merge = True,
                einsum_merge_scalars = False,
                einsum_remove_one = True,
                einsum_pull_removable_axes = False,
                einsum_elim_removable_axes_weak = True,
                einsum_permute_to_rearrange = True,
                einsum_pull_scatter = False,
                einsum_push_down_trace = False,
                einsum_concat_to_add = False,
                # Array (none)
                # Symbol (none)
                # Scalar (none)
                # Add
                remove_add_few_input = True,
                add_flatten = False,
                add_elim_zeros = True,
                add_collapse_scalar_inputs = False,
                add_deduplicate = True,
                add_pull_removable_axes = False,
                add_pull_scatter = False,
                add_pull_diags = False,
                add_fuse_scalar_multiples = False,
                add_elim_removable_axes_weak = True,
                # Rearrange
                rearrange_elim_identity = True,
                rearrange_fuse = True,
                rearrange_merge_scalar = True,
                permute_of_einsum_merge = True,
                # Index
                index_elim_identity = True,
                index_fuse = False,
                index_merge_scalar = True,
                index_einsum_to_scatter = False,
                index_concat_drop_unreached = False,
                # GeneralFunction
                generalfunction_pull_removable_axes = False,
                generalfunction_merge_inverses = False,
                generalfunction_special_case_simplification = False,
                generalfunction_evaluate_simple = False,
                # Concat
                concat_elim_identity = True,
                concat_elim_split = False,
                concat_pull_removable_axes = False,
                concat_merge_uniform = True,
                concat_drop_size_zero = True,
                concat_fuse = True,
                concat_repeat_to_rearrange = True,
                concat_to_scatter = False,
                # Scatter
                scatter_elim_identity = False,
                scatter_fuse = False,
                scatter_pull_removable_axes = False,
                # Conv (none)
                # Module
                elim_empty_module = True,
                elim_no_input_module = True,
                # SetSymbolicShape (none)
                # Tag (none)
                # DiscreteVar (none)
                # StoredCumulantVar (none)
                # Cumulant (none)
            )
        """
    def set(
        self,
        # Einsum
        einsum_elim_zero: Optional[bool] = None,
        einsum_elim_identity: Optional[bool] = None,
        einsum_flatten: Optional[bool] = None,
        einsum_of_permute_merge: Optional[bool] = None,
        einsum_merge_scalars: Optional[bool] = None,
        einsum_remove_one: Optional[bool] = None,
        einsum_pull_removable_axes: Optional[bool] = None,
        einsum_elim_removable_axes_weak: Optional[bool] = None,
        einsum_permute_to_rearrange: Optional[bool] = None,
        einsum_pull_scatter: Optional[bool] = None,
        einsum_push_down_trace: Optional[bool] = None,
        einsum_concat_to_add: Optional[bool] = None,
        # Array (none)
        # Symbol (none)
        # Scalar (none)
        # Add
        remove_add_few_input: Optional[bool] = None,
        add_flatten: Optional[bool] = None,
        add_elim_zeros: Optional[bool] = None,
        add_collapse_scalar_inputs: Optional[bool] = None,
        add_deduplicate: Optional[bool] = None,
        add_pull_removable_axes: Optional[bool] = None,
        add_pull_scatter: Optional[bool] = None,
        add_pull_diags: Optional[bool] = None,
        add_fuse_scalar_multiples: Optional[bool] = None,
        add_elim_removable_axes_weak: Optional[bool] = None,
        # Rearrange
        rearrange_elim_identity: Optional[bool] = None,
        rearrange_fuse: Optional[bool] = None,
        rearrange_merge_scalar: Optional[bool] = None,
        permute_of_einsum_merge: Optional[bool] = None,
        # Index
        index_elim_identity: Optional[bool] = None,
        index_fuse: Optional[bool] = None,
        index_merge_scalar: Optional[bool] = None,
        index_einsum_to_scatter: Optional[bool] = None,
        index_concat_drop_unreached: Optional[bool] = None,
        # GeneralFunction
        generalfunction_pull_removable_axes: Optional[bool] = None,
        generalfunction_merge_inverses: Optional[bool] = None,
        generalfunction_special_case_simplification: Optional[bool] = None,
        generalfunction_evaluate_simple: Optional[bool] = None,
        # Concat
        concat_elim_identity: Optional[bool] = None,
        concat_elim_split: Optional[bool] = None,
        concat_pull_removable_axes: Optional[bool] = None,
        concat_merge_uniform: Optional[bool] = None,
        concat_drop_size_zero: Optional[bool] = None,
        concat_fuse: Optional[bool] = None,
        concat_repeat_to_rearrange: Optional[bool] = None,
        concat_to_scatter: Optional[bool] = None,
        # Scatter
        scatter_elim_identity: Optional[bool] = None,
        scatter_fuse: Optional[bool] = None,
        scatter_pull_removable_axes: Optional[bool] = None,
        # Conv (none)
        # Module
        elim_empty_module: Optional[bool] = None,
        elim_no_input_module: Optional[bool] = None,
        # SetSymbolicShape (none)
        # Tag (none)
        # DiscreteVar (none)
        # StoredCumulantVar (none)
        # Cumulant (none)
    ) -> SimpFnSubset: ...

# TODO: better doc + examples for sampling stuff!

class SampleSpec:
    def sample_var(self, d: DiscreteVar, device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp()) -> Circuit: ...
    def get_empirical_weights(self) -> Circuit: ...
    def get_overall_weight(self) -> Circuit: ...
    def get_sample_shape(self) -> Circuit: ...
    def get_transform(self, device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp()) -> Transform: ...
    def get_sample_expander(
        self,
        var_matcher: IterativeMatcherIn = default_var_matcher(),
        default_fancy_validate: bool = False,
        default_assert_any_found: bool = False,
        suffix: Optional[str] = "sample",
        device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp(),
    ) -> Expander:
        """
        get a sample expander: an Expander configured to sample from vars. It will sample
        from all vars that are matched by var_matcher and batch over them
        together.

        Note: sampling the child of a cumulant doesn't work - use Sampler for
        sampling + estimation. As such, the var_matcher should term
        early at cumulants.

        Note that even if this doesn't match any vars it will always add batch dims.
        """

class OptimizationSettings(object):
    verbose: int
    max_memory: int
    max_single_tensor_memory: int
    max_memory_fallback: Optional[int]
    scheduling_num_mem_chunks: int
    distribute_min_size: Optional[int]
    scheduling_naive: bool
    scheduling_timeout: int
    scheduling_simplify: bool
    adjust_numerical_scale: bool
    numerical_scale_min: float
    numerical_scale_max: float
    capture_and_print: bool
    create_tests: bool
    log_simplifications: bool
    log_slow_einsums: bool
    save_circ: bool
    optimization_parallelism: Optional[int]
    # Warning: keep_all_names will prevent identical nodes with different names from being deduplicated
    # potentially making optimization much worse
    # also there may be bugs where this hangs because it changes names forever (report bug, we'll fix)
    keep_all_names: bool
    push_down_index: bool
    deep_maybe_distribute: bool
    simp_fn_subset: SimpFnSubset
    def __init__(
        # warning: the pyi defaults are not exactly accurate. real values are in circuit_optimizer.rs.
        # those values should be the same as these, but I don't feel like guaranteeing that cuz lazy.
        self,
        verbose=0,
        max_memory=9_000_000_000,
        max_single_tensor_memory=9_000_000_000,
        max_memory_fallback=None,
        scheduling_num_mem_chunks=200,
        distribute_min_size=None,
        scheduling_naive=False,
        scheduling_timeout=5_000,
        scheduling_simplify=True,
        adjust_numerical_scale=False,
        numerical_scale_min=1e-8,
        numerical_scale_max=1e8,
        capture_and_print=False,
        create_tests=False,
        log_simplifications=False,
        log_slow_einsums=False,
        save_circ=False,
        optimization_parallelism=8,
        keep_all_names=False,
        push_down_index: bool = True,
        deep_maybe_distribute: bool = True,
        simp_fn_subset: SimpFnSubset = SimpFnSubset.compiler_default(),
    ) -> None: ...

class OptimizationContext(object):
    settings: OptimizationSettings
    cache: None  # there is a cache, but python cant see it
    def stringify_logs(self) -> str: ...
    @staticmethod
    def new_settings(settings: OptimizationSettings) -> OptimizationContext: ...
    @staticmethod
    def new_settings_circuit(
        settings: OptimizationSettings,
        circuit: Circuit,
    ) -> OptimizationContext: ...

class ExceptionWithRustContext(Exception):
    @property
    def exc(self) -> BaseException: ...

# to generate below exception stubs, `cargo run --bin print_exception_stubs`
class BatchError(ValueError): ...
class BatchRequiresMultipleAxesError(BatchError): ...
class BatchAxisOriginatesTooHighError(BatchError): ...
class BatchNumBatchesIsZeroError(BatchError): ...
class ConstructError(ValueError): ...
class ConstructNotSupportedYetError(ConstructError): ...
class ConstructConvStridePaddingNotFullError(ConstructError): ...
class ConstructArrayHasReservedSymbolicShapeError(ConstructError): ...
class ConstructConvInputWrongShapeError(ConstructError): ...
class ConstructConvFilterWrongShapeError(ConstructError): ...
class ConstructConvFilterMustBeOddLengthError(ConstructError): ...
class ConstructConvStrideMustDivideError(ConstructError): ...
class ConstructConvGroupsMustDivideError(ConstructError): ...
class ConstructConvInputFilterDifferentNumInputChannelsError(ConstructError): ...
class ConstructDiscreteVarNoSamplesDimError(ConstructError): ...
class ConstructDiscreteVarWrongSamplesDimError(ConstructError): ...
class ConstructDiscreteVarProbsMustBe1dError(ConstructError): ...
class ConstructStoredCumulantVarNeedsMeanVarianceError(ConstructError): ...
class ConstructStoredCumulantVarInvalidCumulantNumberError(ConstructError): ...
class ConstructStoredCumulantVarCumulantWrongShapeError(ConstructError): ...
class ConstructEinsumWrongNumChildrenError(ConstructError): ...
class ConstructEinsumLenShapeDifferentFromAxesError(ConstructError): ...
class ConstructEinsumAxisSizeDifferentError(ConstructError): ...
class ConstructEinsumOutputNotSubsetError(ConstructError): ...
class ConstructRearrangeWrongInputShapeError(ConstructError): ...
class ConstructGeneralFunctionWrongInputShapeError(ConstructError): ...
class ConstructGeneralFunctionPyNotInstanceError(ConstructError): ...
class ConstructConcatZeroNodesError(ConstructError): ...
class ConstructConcatShapeDifferentError(ConstructError): ...
class ConstructAxisOutOfBoundsError(ConstructError): ...
class ConstructScatterShapeWrongError(ConstructError): ...
class ConstructScatterIndexTypeUnimplementedError(ConstructError): ...
class ConstructUnknownGeneralFunctionError(ConstructError): ...
class ConstructModuleWrongNumberChildrenError(ConstructError): ...
class ConstructModuleUnknownArgumentError(ConstructError): ...
class ConstructModuleMissingNamesError(ConstructError): ...
class ConstructModuleExtractNotPresentError(ConstructError): ...
class ConstructModuleSomeArgsNamedNoneError(ConstructError): ...
class ConstructModuleArgsDupNamesError(ConstructError): ...
class ConstructModuleSomeArgsNotPresentError(ConstructError): ...
class ConstructModuleExpectedSymbolError(ConstructError): ...
class ConstructModuleExpectedSymbolOnMapError(ConstructError): ...
class ConstructNamedAxisAboveRankError(ConstructError): ...
class ConstructNoEquivalentExplicitlyComputableError(ConstructError): ...
class ConstructUnflattenButNDimNot1Error(ConstructError): ...
class ConstructModulePassedNamePrefixAndUseSelfNameAsPrefixError(ConstructError): ...
class ExpandError(ValueError): ...
class ExpandWrongNumChildrenError(ExpandError): ...
class ExpandBatchingRankTooLowError(ExpandError): ...
class ExpandFixedIndexError(ExpandError): ...
class ExpandConcatAxisError(ExpandError): ...
class ExpandGeneralFunctionTriedToBatchNonBatchableInputError(ExpandError): ...
class ExpandNodeUnhandledVariantError(ExpandError): ...
class ExpandCumulantRankChangedError(ExpandError): ...
class ExpandModuleArgSpecSymbolChangedInExpandError(ExpandError): ...
class ExpandModuleRankReducedError(ExpandError): ...
class ExpandModuleTriedToBatchUnbatchableInputError(ExpandError): ...
class ExpandModuleTriedToExpandUnexpandableInputError(ExpandError): ...
class ExpandModuleTriedToExpandOnNonSymbolicSizeAndBannedError(ExpandError): ...
class SubstitutionError(ValueError): ...
class SubstitutionCircuitHasFreeSymsBoundByNestedModuleError(SubstitutionError): ...
class SubstitutionFoundNEQFreeSymbolWithSameIdentificationError(SubstitutionError): ...
class MiscInputError(ValueError): ...
class MiscInputNotBroadcastableError(MiscInputError): ...
class MiscInputIndexDtypeNotI64Error(MiscInputError): ...
class MiscInputItemOutOfBoundsError(MiscInputError): ...
class MiscInputChildrenMultipleDtypesError(MiscInputError): ...
class MiscInputCastIncompatibleDeviceDtypeError(MiscInputError): ...
class MiscInputChildrenMultipleDevicesError(MiscInputError): ...
class ParseError(ValueError): ...
class ParseInvalidNumberError(ParseError): ...
class ParseInvalidIndexError(ParseError): ...
class ParseInvalidUuidError(ParseError): ...
class ParseEinsumStringInvalidError(ParseError): ...
class ParseEinsumStringNoArrowError(ParseError): ...
class ParseFactorProductTooLargeError(ParseError): ...
class ParseSymbolicSizeNumberOutOfBoundsError(ParseError): ...
class ParseCircuitError(ParseError): ...
class ParseCircuitRegexDidntMatchError(ParseCircuitError): ...
class ParseCircuitRegexDidntMatchGroupError(ParseCircuitError): ...
class ParseCircuitLessIndentationThanFirstItemError(ParseCircuitError): ...
class ParseCircuitInvalidIndentationError(ParseCircuitError): ...
class ParseCircuitExpectedNoExtraInfoError(ParseCircuitError): ...
class ParseCircuitCycleError(ParseCircuitError): ...
class ParseCircuitWrongNumberChildrenError(ParseCircuitError): ...
class ParseCircuitUnexpectedChildInfoError(ParseCircuitError): ...
class ParseCircuitInvalidVariantError(ParseCircuitError): ...
class ParseCircuitShapeNeededError(ParseCircuitError): ...
class ParseCircuitExpectedOneCircuitGotMultipleError(ParseCircuitError): ...
class ParseCircuitInvalidTerseBoolError(ParseCircuitError): ...
class ParseCircuitReferenceCircuitHasChildrenError(ParseCircuitError): ...
class ParseCircuitRepeatedCircuitHasChildrenError(ParseCircuitError): ...
class ParseCircuitHasParentInfoButNoParentError(ParseCircuitError): ...
class ParseCircuitReferenceCircuitNameFollowedByAdditionalInfoError(ParseCircuitError): ...
class ParseCircuitPassedInShapeDoesntMatchComputedShapeError(ParseCircuitError): ...
class ParseCircuitArrayShapeLoadedFromHashDiffersFromProvidedShapeError(ParseCircuitError): ...
class ParseCircuitOnCircuitRepeatInfoIsNotSameError(ParseCircuitError): ...
class ParseCircuitModuleNoSpecCircuitError(ParseCircuitError): ...
class ParseCircuitModuleRequiresInputChildToHaveParentInfoError(ParseCircuitError): ...
class ParseCircuitStoredCumulantVarExtraInvalidError(ParseCircuitError): ...
class ParseCircuitInvalidAxisShapeFormatError(ParseCircuitError): ...
class RearrangeSpecError(ValueError): ...
class RearrangeSpecInputNotConformableError(RearrangeSpecError): ...
class RearrangeSpecHasWildcardSizesError(RearrangeSpecError): ...
class RearrangeSpecNotConvertableError(RearrangeSpecError): ...
class RearrangeSpecTooManyWildcardSizesError(RearrangeSpecError): ...
class RearrangeSpecAxesAndCountsDontMatchError(RearrangeSpecError): ...
class RearrangeSpecNotFusableError(RearrangeSpecError): ...
class RearrangeSpecAmbiguousExpandError(RearrangeSpecError): ...
class RearrangeSpecRepeatedAxesError(RearrangeSpecError): ...
class RearrangeSpecIntsNotUniqueError(RearrangeSpecError): ...
class RearrangeSpecInpNotInOutError(RearrangeSpecError): ...
class RearrangeSpecIntNotInSizesError(RearrangeSpecError): ...
class RearrangeSpecIntOnlyInOutputWithoutSizeError(RearrangeSpecError): ...
class RearrangeSpecInputAxisHasMultipleWildCardsError(RearrangeSpecError): ...
class RearrangeSpecLenShapeTooLargeError(RearrangeSpecError): ...
class RearrangeSpecAxesToCombineNotSubsetError(RearrangeSpecError): ...
class RearrangeSpecCantUnflattenScalarError(RearrangeSpecError): ...
class PermError(ValueError): ...
class PermIntsNotUniqueError(PermError): ...
class PermNotContiguousIntsError(PermError): ...
class RearrangeParseError(ParseError): ...
class RearrangeParseArrowIssueError(RearrangeParseError): ...
class RearrangeParseFailedToMatchRegexError(RearrangeParseError): ...
class RearrangeParseTooManyAxesError(RearrangeParseError): ...
class SchedulingError(ValueError): ...
class SchedulingNotExplicitlyComputableError(SchedulingError): ...
class SchedulingOOMError(ValueError): ...
class SchedulingOOMManyError(SchedulingOOMError): ...
class SchedulingOOMSingleError(SchedulingOOMError): ...
class SchedulingOOMSimpError(SchedulingOOMError): ...
class SchedulingOOMThreadsLostError(SchedulingOOMError): ...
class SchedulingOOMExhaustiveTimeoutError(SchedulingOOMError): ...
class TensorEvalError(ValueError): ...
class TensorEvalNotExplicitlyComputableInternalError(TensorEvalError): ...
class TensorEvalNotExplicitlyComputableError(TensorEvalError): ...
class TensorEvalNotConstantError(TensorEvalError): ...
class TensorEvalModulesCantBeDirectlyEvalutedInternalError(TensorEvalError): ...
class TensorEvalDeviceDtypeErrorError(TensorEvalError): ...
class SampleError(ValueError): ...
class SampleUnhandledVarErrorError(SampleError): ...
class SampleGroupWithIncorrectNdimError(SampleError): ...
class SampleDifferentNumSubsetsThanGroupsError(SampleError): ...
class IndexError(ValueError): ...
class IndexTensorNDimNot1Error(IndexError): ...
class IndexIndexOutOfBoundsError(IndexError): ...
class IndexIndexRankTooHighError(IndexError): ...
class IndexUnsupportedSliceError(IndexError): ...
class NestError(ValueError): ...
class NestMultipleRestError(NestError): ...
class NestMatchersOverlapError(NestError): ...
class NestMatcherOverlapsWithExplicitIntsError(NestError): ...
class NestMatcherMatchedMultipleAndMustBeUniqueError(NestError): ...
class NestMatcherMatchedNoneAndMustExistError(NestError): ...
class NestTraversalMatchedNothingError(NestError): ...
class NestIntNotContainedInRangeCountError(NestError): ...
class NestPermHasWrongLenError(NestError): ...
class NestOrigNumPermWhenNotPresentInOrigError(NestError): ...
class NestPermutationMissesIdxsAndNoRestInSpecError(NestError): ...
class PushDownIndexError(ValueError): ...
class PushDownIndexNoopOnConcatError(PushDownIndexError): ...
class PushDownIndexNoopOnGeneralFunctionError(PushDownIndexError): ...
class PushDownIndexGeneralFunctionSomeAxesNotPossibleError(PushDownIndexError): ...
class PushDownIndexModuleSomeAxesNotPossibleError(PushDownIndexError): ...
class PushDownIndexEinsumNoopError(PushDownIndexError): ...
class PushDownIndexEinsumSomeAxesNotPossibleError(PushDownIndexError): ...
class PushDownIndexRearrangeNotPossibleError(PushDownIndexError): ...
class PushDownIndexScatterNoopError(PushDownIndexError): ...
class PushDownIndexScatterSomeAxesNotPossibleError(PushDownIndexError): ...
class PushDownIndexThroughIndexError(PushDownIndexError): ...
class PushDownIndexUnimplementedTypeError(PushDownIndexError): ...
class PushDownIndexConcatSplitSectionsTensorUnsupportedError(PushDownIndexError): ...
class DistributeError(ValueError): ...
class DistributeCouldntBroadcastError(DistributeError): ...
class DistributeOperandIsNotAddError(DistributeError): ...
class DistributeOperandIdxTooLargeError(DistributeError): ...
class DistributeEmptyAddUnsupportedError(DistributeError): ...
class DistributeNoopError(DistributeError): ...
class SimpConfigError(ValueError): ...
class SimpConfigFnNamesNotValidError(SimpConfigError): ...
class CumulantRewriteError(ValueError): ...
class CumulantRewriteNodeToExpandNotFoundError(CumulantRewriteError): ...
class CumulantRewriteUnexpandableTypeError(CumulantRewriteError): ...
class CumulantRewriteUnimplementedTypeError(CumulantRewriteError): ...
class GeneralFunctionShapeError(ValueError): ...
class GeneralFunctionShapeWrongNumShapesError(GeneralFunctionShapeError): ...
class GeneralFunctionShapeNDimTooSmallError(GeneralFunctionShapeError): ...
class GeneralFunctionShapeIndexShapeInvalidError(GeneralFunctionShapeError): ...
class ReferenceCircError(ValueError): ...
class ReferenceCircByNameHasNoneNameError(ReferenceCircError): ...
class ReferenceCircDuplicateIdentifierError(ReferenceCircError): ...
class SymbolicSizeOverflowError(OverflowError): ...
class SymbolicSizeOverflowProductTooLargeError(SymbolicSizeOverflowError): ...
class SymbolicSizeOverflowSymbolicSizeNumberOutOfBoundsError(SymbolicSizeOverflowError): ...
class SymbolicSizeSetError(OverflowError): ...
class SymbolicSizeSetTriedToSetNonZeroToZeroError(SymbolicSizeSetError): ...
class SymbolicSizeSetFactorDoesntDivideSetToError(SymbolicSizeSetError): ...
class SymbolicSizeSetNoSymbolicAndSizesNotEqualError(SymbolicSizeSetError): ...
class SymbolicSizeSetFailedToSatisfyContraintsError(SymbolicSizeSetError): ...
class IterativeMatcherError(ValueError): ...
class IterativeMatcherNumUpdatedMatchersNEQToNumChildrenError(IterativeMatcherError): ...
class IterativeMatcherOperationDoesntSupportArgPerChildError(IterativeMatcherError): ...
class IterativeMatcherChildNumbersOutOfBoundsError(IterativeMatcherError): ...
class ModuleBindError(RuntimeError): ...
class ModuleBindExpectedSymbolError(ModuleBindError): ...
class NamedAxesError(RuntimeError): ...
class NamedAxesForbiddenCharacterError(NamedAxesError): ...
class PushDownModuleError(ValueError): ...
class PushDownModulePushPastPreviouslyBoundSymbolError(PushDownModuleError): ...
class PushDownModulePushingPastModuleWhichOverridesSymError(PushDownModuleError): ...
class ExtractSymbolsError(ValueError): ...
class ExtractSymbolsGetFoundNonSymbolError(ExtractSymbolsError): ...
class ExtractSymbolsBatchedInputError(ExtractSymbolsError): ...
class ExtractSymbolsArgSpecInconsistentError(ExtractSymbolsError): ...
class ExtractSymbolsBoundInputInconsistentError(ExtractSymbolsError): ...
class ExtractSymbolsHasBindingsFromOuterModuleError(ExtractSymbolsError): ...

class SetOfCircuitIdentities(object):
    """Set of hashes of circuits, can check for inclusion but not extract circuits themselves.
    this is a last resort when you need to keep track of more circuits than python can handle
    this is painful
    """

    def __init__(self) -> None: ...
    def __contains__(self, circuit: Circuit) -> bool: ...
    def insert(self, circuit: Circuit): ...
    def extend(self, other: SetOfCircuitIdentities): ...
    def union(self, other: SetOfCircuitIdentities) -> SetOfCircuitIdentities: ...
    def intersection(self, other: SetOfCircuitIdentities) -> SetOfCircuitIdentities: ...

class Schedule(object):
    @property
    def device_dtype(self) -> TorchDeviceDtype: ...
    @property
    def constants(self) -> Dict[int, IrreducibleNode]: ...
    @property
    def scalars(self) -> Dict[int, Scalar]: ...
    @property
    def instructions(self) -> List[Union[int, Tuple[int, Circuit]]]: ...
    def validate(self, validate_output: bool) -> None: ...
    def replace_tensors(self, map: Dict[bytes, torch.Tensor], allow_missing: bool = False) -> Schedule: ...
    def map_tensors(self, f: Callable[[bytes], Optional[torch.Tensor]]) -> Schedule: ...
    def get_stats(self) -> ScheduleStats: ...
    def evaluate(self, settings: OptimizationSettings = OptimizationSettings()) -> torch.Tensor: ...
    def evaluate_many(self, settings: OptimizationSettings = OptimizationSettings()) -> List[torch.Tensor]: ...
    def serialize(self) -> str: ...
    @staticmethod
    def deserialize(string: str, device: str, tensor_cache: Optional[TensorCacheRrfs] = None) -> Schedule: ...
    def evaluate_remote(self, url: str) -> torch.Tensor: ...
    def evaluate_remote_many(self, url: str) -> List[torch.Tensor]: ...
    def tosend(self) -> ScheduleToSend: ...

class ScheduleStats(object):
    @property
    def max_mem(self) -> int: ...
    @property
    def constant_mem(self) -> int: ...
    @property
    def max_circuit_set(self) -> Set[Circuit]: ...

class ScheduleToSend(object):
    """this is just used to detach schedule from thread before sending to remote worker"""

    def evaluate_remote(self, url: str, device: str) -> torch.Tensor: ...
    def evaluate_remote_many(self, url: str, device: str) -> List[torch.Tensor]: ...

class Regex:
    r"""
    Unlike in normal regex, by default `.` symbols are escaped and so match actual `.` symbols in names.
    `\.` can be used for the regex wildcard character. Alternately, you can set escape_dot = False to disable this
    (so `.` is the wildcard and `\.` matches periods).

    Also doesn't require a full string match. You can add `^` and `$` to the start and end of your string do so.
    """
    @property
    def pattern(self) -> str: ...
    @property
    def escape_dot(self) -> bool: ...
    def __init__(self, pattern: str, escape_dot: bool = True) -> None: ...
    def call(self, s: str) -> bool: ...

MatcherIn = Union[
    bool,
    str,
    set[str],
    frozenset[str],
    Type[Circuit],
    set[Type[Circuit]],
    frozenset[Type[Circuit]],
    Circuit,
    set[Circuit],
    frozenset[Circuit],
    Regex,
    BoundAnyFound,
    Matcher,
    Callable[[Circuit], bool],
]

class Matcher(object):
    """
    Matchers just return a yes/no of whether the input circuit matches.
    They can be matchers to names, or any property of circuits (see methods)
    """

    def __init__(self, *inps: MatcherIn) -> None:
        """Does 'any' of the inputs if there are multiple"""
    def __call__(self, circuit: Circuit) -> bool:
        """Returns true if the root node matches. Also see .are_any_found()"""
    def call(self, circuit: Circuit) -> bool: ...
    def debug_print_to_str(self) -> str: ...
    def get_first(self, circuit: Circuit) -> Optional[Circuit]: ...
    @staticmethod
    def true_matcher() -> Matcher: ...
    @staticmethod
    def false_matcher() -> Matcher: ...
    @staticmethod
    def regex(pattern: str, escape_dot: bool = True) -> Matcher: ...
    @staticmethod
    def match_any_found(finder: IterativeMatcherIn) -> Matcher:
        """alias for Matcher(IterativeMatcher(finder).any_found())"""
    @staticmethod
    def match_any_child_found(finder: IterativeMatcherIn) -> Matcher:
        """alias for restrict( Matcher.match_any_found(IterativeMatcher(finder),start_depth=1))"""
    @staticmethod
    def types(*types: Type[Circuit]) -> Matcher: ...
    @staticmethod
    def circuits(*circuits: Circuit) -> Matcher: ...
    @staticmethod
    def all(*matchers: MatcherIn) -> Matcher: ...
    @staticmethod
    def any(*matchers: MatcherIn) -> Matcher: ...
    def new_not(self) -> Matcher: ...
    def new_and(self, *others: MatcherIn) -> Matcher: ...
    def new_or(self, *others: MatcherIn) -> Matcher: ...
    def __invert__(self) -> Matcher: ...
    def __and__(self, other: MatcherIn) -> Matcher: ...
    def __or__(self, other: MatcherIn) -> Matcher: ...
    def __rand__(self, other: MatcherIn) -> Matcher: ...
    def __ror__(self, other: MatcherIn) -> Matcher: ...
    def to_iterative_matcher(self) -> IterativeMatcher: ...
    # below are methods which work via first converting to iterative matcher
    # and then calling the matching method on iterative matcher
    def chain(self, *rest: IterativeMatcherIn, must_be_sub: bool = False) -> IterativeMatcher:
        """
        Constructs chain [self, *rest]

        For finding paths that satisfy a succession of matchers.

        A node is considered matched if there is a length 1 chain containing a
        matcher that matches the node.

        On a single iteration, this returns a new matcher as follows:
          - we want to construct the chains that will become the chains of the new matcher
          - for each chain, see whether the first matcher in the chain matches the node
            - if so, it is considered satisfied
              - the tail of this chain will now itself be one of the new chains
              - if must_be_sub is False for the next item in this chain, it means the same node is
                allowed to satisfy it, so we run again with the same node and the tail of this chain.
              - if must_be_sub is True, we don't run the chain again. In
                practice, this is just implemented by wrapping all items in the
                chain other than the first item with `.filter(start_depth=1)`
            - whether or not it matches, it returns a new matcher, and we construct a new chain
              with this new matcher as the head, and the old tail as the tail

        Generally you will probably use `new_chain` or `chain` to construct an instance with a single chain, say
        (A, B, C); but during iteration (e.g. in a Getter) new matchers will be
        constructed with multiple chains, e.g. after A matches a node the new matcher will have chains
        ((A', B, C), (B, C)). It's possible B will also match this same node, in
        which case the new matcher will have chains
        ((A', B, C), (B, C), (B', C), (C,)). If this is then applied to a node
        that matches C, that node will be matched, since chains contains the
        chain (C,).
        """
    def chain_many(self, *rest: Sequence[IterativeMatcherIn], must_be_sub: bool = False) -> IterativeMatcher:
        """
        Constructs chains [[self, *r] for r in rest]

        For finding paths that satisfy a succession of matchers.

        A node is considered matched if there is a length 1 chain containing a
        matcher that matches the node.

        On a single iteration, this returns a new matcher as follows:
          - we want to construct the chains that will become the chains of the new matcher
          - for each chain, see whether the first matcher in the chain matches the node
            - if so, it is considered satisfied
              - the tail of this chain will now itself be one of the new chains
              - if must_be_sub is False for the next item in this chain, it means the same node is
                allowed to satisfy it, so we run again with the same node and the tail of this chain.
              - if must_be_sub is True, we don't run the chain again. In
                practice, this is just implemented by wrapping all items in the
                chain other than the first item with `.filter(start_depth=1)`
            - whether or not it matches, it returns a new matcher, and we construct a new chain
              with this new matcher as the head, and the old tail as the tail

        Generally you will probably use `new_chain` or `chain` to construct an instance with a single chain, say
        (A, B, C); but during iteration (e.g. in a Getter) new matchers will be
        constructed with multiple chains, e.g. after A matches a node the new matcher will have chains
        ((A', B, C), (B, C)). It's possible B will also match this same node, in
        which case the new matcher will have chains
        ((A', B, C), (B, C), (B', C), (C,)). If this is then applied to a node
        that matches C, that node will be matched, since chains contains the
        chain (C,).
        """
    def get(self, circuit: Circuit, fancy_validate: bool = False) -> Set[Circuit]:
        """Returns the set of all nodes within the circuit which match."""
    def get_unique_op(self, circuit: Circuit, fancy_validate: bool = False) -> Optional[Circuit]: ...
    def get_unique(self, circuit: Circuit, fancy_validate: bool = False) -> Circuit: ...
    def get_paths(self, circuit: Circuit) -> Dict[Circuit, Path]:
        """Return a dict where each matched node is associated with one arbitrary path to it."""
    def get_all_paths(self, circuit: Circuit) -> Dict[Circuit, Paths]:
        """Return a dict where each matched node is associated with a list of every path to it."""
    def get_all_circuits_in_paths(self, circuit: Circuit) -> List[Circuit]: ...
    def module_bind_get(self, circuit: Circuit, fancy_validate: bool = False) -> Set[Circuit]: ...
    def module_bind_get_unique_op(self, circuit: Circuit, fancy_validate: bool = False) -> Optional[Circuit]: ...
    def module_bind_get_unique(self, circuit: Circuit, fancy_validate: bool = False) -> Circuit: ...
    def validate(self, circuit: Circuit) -> None: ...
    def getter(self, default_fancy_validate: bool = False) -> BoundGetter: ...
    def are_any_found(self, circuit: Circuit) -> bool:
        """Returns true if any node within the circuit matches."""
    def any_found(self) -> BoundAnyFound: ...
    def update(
        self,
        circuit: Circuit,
        transform: TransformIn,
        cache_transform: bool = True,
        cache_update: bool = True,
        fancy_validate: bool = False,
        assert_any_found: bool = False,
    ) -> Circuit: ...
    def updater(
        self,
        transform: TransformIn,
        cache_transform: bool = True,
        cache_update: bool = True,
        default_fancy_validate: bool = False,
        default_assert_any_found: bool = False,
    ) -> BoundUpdater: ...

class Finished: ...

FINISHED = Finished()

IterativeMatcherIn = Union[MatcherIn, IterativeMatcher]

class IterativeMatcher:
    """
    Returns IterateMatchResults when called (see docs for that). Useful for iteratively searching
    through a circuit for some matching nodes.
    """

    def __init__(self, *inps: IterativeMatcherIn) -> None:
        """Does 'any' of the inputs if there arejo multiple"""
    def match_iterate(self, circuit: Circuit) -> IterateMatchResults: ...
    def debug_print_to_str(self) -> str: ...
    def validate_matched(self, matched: Set[Circuit]) -> None: ...
    @staticmethod
    def noop_traversal() -> IterativeMatcher: ...
    @staticmethod
    def term(match_next: bool = False) -> IterativeMatcher: ...
    def chain(self, *rest: IterativeMatcherIn, must_be_sub: bool = False) -> IterativeMatcher:
        """
        Constructs chain [self, *rest]

        For finding paths that satisfy a succession of matchers.

        A node is considered matched if there is a length 1 chain containing a
        matcher that matches the node.

        On a single iteration, this returns a new matcher as follows:
          - we want to construct the chains that will become the chains of the new matcher
          - for each chain, see whether the first matcher in the chain matches the node
            - if so, it is considered satisfied
              - the tail of this chain will now itself be one of the new chains
              - if must_be_sub is False for the next item in this chain, it means the same node is
                allowed to satisfy it, so we run again with the same node and the tail of this chain.
              - if must_be_sub is True, we don't run the chain again. In
                practice, this is just implemented by wrapping all items in the
                chain other than the first item with `.filter(start_depth=1)`
            - whether or not it matches, it returns a new matcher, and we construct a new chain
              with this new matcher as the head, and the old tail as the tail

        Generally you will probably use `new_chain` or `chain` to construct an instance with a single chain, say
        (A, B, C); but during iteration (e.g. in a Getter) new matchers will be
        constructed with multiple chains, e.g. after A matches a node the new matcher will have chains
        ((A', B, C), (B, C)). It's possible B will also match this same node, in
        which case the new matcher will have chains
        ((A', B, C), (B, C), (B', C), (C,)). If this is then applied to a node
        that matches C, that node will be matched, since chains contains the
        chain (C,).
        """
    def chain_many(self, *rest: Sequence[IterativeMatcherIn], must_be_sub: bool = False) -> IterativeMatcher:
        """
        Constructs chains [[self, *r] for r in rest]

        For finding paths that satisfy a succession of matchers.

        A node is considered matched if there is a length 1 chain containing a
        matcher that matches the node.

        On a single iteration, this returns a new matcher as follows:
          - we want to construct the chains that will become the chains of the new matcher
          - for each chain, see whether the first matcher in the chain matches the node
            - if so, it is considered satisfied
              - the tail of this chain will now itself be one of the new chains
              - if must_be_sub is False for the next item in this chain, it means the same node is
                allowed to satisfy it, so we run again with the same node and the tail of this chain.
              - if must_be_sub is True, we don't run the chain again. In
                practice, this is just implemented by wrapping all items in the
                chain other than the first item with `.filter(start_depth=1)`
            - whether or not it matches, it returns a new matcher, and we construct a new chain
              with this new matcher as the head, and the old tail as the tail

        Generally you will probably use `new_chain` or `chain` to construct an instance with a single chain, say
        (A, B, C); but during iteration (e.g. in a Getter) new matchers will be
        constructed with multiple chains, e.g. after A matches a node the new matcher will have chains
        ((A', B, C), (B, C)). It's possible B will also match this same node, in
        which case the new matcher will have chains
        ((A', B, C), (B, C), (B', C), (C,)). If this is then applied to a node
        that matches C, that node will be matched, since chains contains the
        chain (C,).
        """
    def children_matcher(self, child_numbers: Set[int]) -> IterativeMatcher:
        """if self matchs, then all of the child numbers provided are matched"""
    def module_arg_matcher(self, arg_sym_matcher: MatcherIn) -> IterativeMatcher:
        """if self matches, then all of the module's arguments whose argument symbols match
        'arg_sym_matcher' are matched"""
    def spec_circuit_matcher(self) -> IterativeMatcher:
        """Like `children_matcher`, but specifically matches on spec_circuit for a module"""
    def filter_module_spec(self, enable: bool = True) -> IterativeMatcher:
        """Avoids matching on circuits belonging to the 'spec' of a module: the spec_circuit + the arg spec symbols"""
    @staticmethod
    def any(*matchers: IterativeMatcherIn) -> IterativeMatcher: ...
    @staticmethod
    def all(*matchers: IterativeMatcherIn) -> IterativeMatcher: ...
    @staticmethod
    def new_chain(
        first: IterativeMatcherIn, *matchers: IterativeMatcherIn, must_be_sub: bool = False
    ) -> IterativeMatcher:
        """
        equivalent to first.chain(*matchers)

        For finding paths that satisfy a succession of matchers.

        A node is considered matched if there is a length 1 chain containing a
        matcher that matches the node.

        On a single iteration, this returns a new matcher as follows:
          - we want to construct the chains that will become the chains of the new matcher
          - for each chain, see whether the first matcher in the chain matches the node
            - if so, it is considered satisfied
              - the tail of this chain will now itself be one of the new chains
              - if must_be_sub is False for the next item in this chain, it means the same node is
                allowed to satisfy it, so we run again with the same node and the tail of this chain.
              - if must_be_sub is True, we don't run the chain again. In
                practice, this is just implemented by wrapping all items in the
                chain other than the first item with `.filter(start_depth=1)`
            - whether or not it matches, it returns a new matcher, and we construct a new chain
              with this new matcher as the head, and the old tail as the tail

        Generally you will probably use `new_chain` or `chain` to construct an instance with a single chain, say
        (A, B, C); but during iteration (e.g. in a Getter) new matchers will be
        constructed with multiple chains, e.g. after A matches a node the new matcher will have chains
        ((A', B, C), (B, C)). It's possible B will also match this same node, in
        which case the new matcher will have chains
        ((A', B, C), (B, C), (B', C), (C,)). If this is then applied to a node
        that matches C, that node will be matched, since chains contains the
        chain (C,).
        """
    @staticmethod
    def new_chain_many(*matchers: Sequence[IterativeMatcherIn], must_be_sub: bool = False) -> IterativeMatcher:
        """each chain must be non-empty

        For finding paths that satisfy a succession of matchers.

        A node is considered matched if there is a length 1 chain containing a
        matcher that matches the node.

        On a single iteration, this returns a new matcher as follows:
          - we want to construct the chains that will become the chains of the new matcher
          - for each chain, see whether the first matcher in the chain matches the node
            - if so, it is considered satisfied
              - the tail of this chain will now itself be one of the new chains
              - if must_be_sub is False for the next item in this chain, it means the same node is
                allowed to satisfy it, so we run again with the same node and the tail of this chain.
              - if must_be_sub is True, we don't run the chain again. In
                practice, this is just implemented by wrapping all items in the
                chain other than the first item with `.filter(start_depth=1)`
            - whether or not it matches, it returns a new matcher, and we construct a new chain
              with this new matcher as the head, and the old tail as the tail

        Generally you will probably use `new_chain` or `chain` to construct an instance with a single chain, say
        (A, B, C); but during iteration (e.g. in a Getter) new matchers will be
        constructed with multiple chains, e.g. after A matches a node the new matcher will have chains
        ((A', B, C), (B, C)). It's possible B will also match this same node, in
        which case the new matcher will have chains
        ((A', B, C), (B, C), (B', C), (C,)). If this is then applied to a node
        that matches C, that node will be matched, since chains contains the
        chain (C,).
        """
    @staticmethod
    def new_children_matcher(
        first_match: IterativeMatcherIn, child_numbers: Union[Set[int], FrozenSet[int]]
    ) -> IterativeMatcher:
        """first_match matchs, then all of the child numbers provided are matched"""
    @staticmethod
    def new_module_arg_matcher(module_matcher: IterativeMatcherIn, arg_sym_matcher: MatcherIn) -> IterativeMatcher:
        """if 'module_matcher' matches, then all of the module's arguments whose argument symbols
        match 'arg_sym_matcher' are matched"""
    @staticmethod
    def new_spec_circuit_matcher(first_match: IterativeMatcherIn) -> IterativeMatcher:
        """Like `new_children_matcher`, but specfically matches on spec_circuit for a module"""
    @staticmethod
    def new_func(f: Callable[[Circuit], IterateMatchResults]) -> IterativeMatcher: ...
    def new_or(self, *others: IterativeMatcherIn) -> IterativeMatcher: ...
    def __or__(self, other: IterativeMatcherIn) -> IterativeMatcher: ...
    def __ror__(self, other: IterativeMatcherIn) -> IterativeMatcher: ...
    def __and__(self, other: IterativeMatcherIn) -> IterativeMatcher: ...
    def __rand__(self, other: IterativeMatcherIn) -> IterativeMatcher: ...
    def get(self, circuit: Circuit, fancy_validate: bool = False) -> Set[Circuit]:
        """Returns the set of all nodes within the circuit which match."""
    def get_unique_op(self, circuit: Circuit, fancy_validate: bool = False) -> Optional[Circuit]: ...
    def get_unique(self, circuit: Circuit, fancy_validate: bool = False) -> Circuit: ...
    def get_paths(self, circuit: Circuit) -> Dict[Circuit, Path]:
        """Return a dict where each matched node is associated with one arbitrary path to it."""
    def get_all_paths(self, circuit: Circuit) -> Dict[Circuit, Paths]:
        """Return a dict where each matched node is associated with a list of every path to it."""
    def validate(self, circuit: Circuit) -> None: ...
    def getter(self, default_fancy_validate: bool = False) -> BoundGetter: ...
    def are_any_found(self, circuit: Circuit) -> bool:
        """Returns true if any node within the circuit matches."""
    def any_found(self) -> BoundAnyFound: ...
    def update(
        self,
        circuit: Circuit,
        transform: TransformIn,
        cache_transform: bool = True,
        cache_update: bool = True,
        fancy_validate: bool = False,
        assert_any_found: bool = False,
    ) -> Circuit: ...
    def updater(
        self,
        transform: TransformIn,
        cache_transform: bool = True,
        cache_update: bool = True,
        default_fancy_validate: bool = False,
        default_assert_any_found: bool = False,
    ) -> BoundUpdater: ...
    def apply_in_traversal(self, circuit: Circuit, transform: Transform) -> Circuit:
        """
        Replace everything outside traversal with symbols,
        then apply function,
        then replace back"""

def restrict(
    matcher: IterativeMatcherIn,
    term_if_matches: bool = False,
    start_depth: Optional[int] = None,
    end_depth: Optional[int] = None,
    term_early_at: MatcherIn = False,
) -> IterativeMatcher:
    """
    Helper with some basic rules you may want to use to control your node matching iterations.

    term_if_matches : bool
        if true, stops once it has found a match
    start_depth : Optional[int]
        depth at which to start matching
    end_depth : Optional[int]
        stops at this depth (e.g., exclusively match before this depth)
    term_early_at : Matcher
        specifies nodes at which to stop iteration

    Note that if A and B are both iterative matchers then
        - A.chain(restrict(B, term_early_at="c"))
        - restrict(A.chain(B), term_early_at="c")
        - restrict(A, term_early_at="c").chain(B)
    Are all meaningful but distinct iterative matchers.
    """

def restrict_sl(
    matcher: IterativeMatcherIn,
    term_if_matches: bool = False,
    depth_slice: slice = slice(None),
    term_early_at: MatcherIn = False,
) -> IterativeMatcher:
    """
    Helper with some basic rules you may want to use to control your node matching iterations.

    term_if_matches : bool
        if true, stops once it has found a match
    depth_slice : slice
        depth range at which to start and stop matching
        (you might find SLICER useful for this)
    term_early_at : Matcher
        specifies nodes at which to stop iteration"""

def new_traversal(
    start_depth: Optional[int] = None,
    end_depth: Optional[int] = None,
    term_early_at: MatcherIn = False,
) -> IterativeMatcher:
    """
    Create a new iterative matcher that matches everything except for the restrictions specified.

    start_depth : Optional[int]
        depth at which to start matching
    end_depth : Optional[int]
        stops at this depth (e.g., exclusively match before this depth)
    term_early_at : Matcher
        specifies nodes at which to stop iteration"""

def print_matcher_debug(circuit: Circuit, matcher: IterativeMatcher, print_options: PrintOptions = PrintOptions()): ...
def repr_matcher_debug(
    circuit: Circuit, matcher: IterativeMatcher, print_options: PrintOptions = PrintOptions()
) -> str: ...
def append_matchers_to_names(circuit: Circuit, matcher: IterativeMatcher, discard_old_name=False) -> Circuit: ...

UpdatedIterativeMatcherIn = Union[Finished, IterativeMatcherIn, List[Union[Finished, IterativeMatcherIn]]]
"""
See docs for IterateMatchResults
(TODO: maybe we should add some pyfunctions for operating on these?)
"""
UpdatedIterativeMatcher = Union[Finished, IterativeMatcher, List[Union[Finished, IterativeMatcher]]]
"""
See docs for IterateMatchResults
(TODO: maybe we should add some pyfunctions for operating on these?)
"""

class IterateMatchResults:
    """
    updated: new matcher to be applied to children
    - if None, use the same matcher
    - if Finished, terminate here (effectively apply False matcher to all children and terminate traveral
    - if a list, we do the use the corresponding new matcher/finished for each
      child respectively (the number of elements in the list must correspond to
      the number of children for the given circuit)

    found: did we match this node?
    """

    updated: Optional[UpdatedIterativeMatcher]
    found: bool
    def __init__(self, updated: Optional[UpdatedIterativeMatcherIn] = None, found: bool = False) -> None: ...
    @staticmethod
    def new_finished(found: bool = False) -> IterateMatchResults: ...
    def to_tup(self) -> Tuple[Optional[UpdatedIterativeMatcher], bool]: ...
    def unwrap_or_same(self, matcher: IterativeMatcher) -> Tuple[UpdatedIterativeMatcher, bool]: ...

TransformIn = Union[Callable[[Circuit], Circuit], Transform]

class Transform:
    def __init__(self, inp: TransformIn) -> None: ...
    def run(self, circuit: Circuit) -> Circuit: ...
    def __call__(self, circuit: Circuit) -> Circuit:
        """Alias for run"""
    def debug_print_to_str(self) -> str: ...
    @staticmethod
    def ident() -> Transform: ...
    def updater(
        self,
        cache_transform: bool = True,
        cache_update: bool = True,
    ) -> Updater: ...

class Updater:
    default_fancy_validate: bool
    def __init__(
        self,
        transform: TransformIn,
        cache_transform: bool = True,
        cache_update: bool = True,
        default_fancy_validate: bool = False,
        default_assert_any_found: bool = False,
    ) -> None: ...
    @property
    def transform(self) -> Transform: ...
    @property
    def cache_transform(self) -> bool: ...
    @property
    def cache_update(self) -> bool: ...
    def update(
        self,
        circuit: Circuit,
        matcher: IterativeMatcherIn,
        fancy_validate: Optional[bool] = None,
        assert_any_found: Optional[bool] = None,
    ) -> Circuit: ...
    def __call__(
        self,
        circuit: Circuit,
        matcher: IterativeMatcherIn,
        fancy_validate: Optional[bool] = None,
        assert_any_found: Optional[bool] = None,
    ) -> Circuit:
        """Alias for update"""
    def bind(self, matcher: IterativeMatcherIn) -> BoundUpdater: ...

class BoundUpdater:
    updater: Updater
    @property
    def matcher(self) -> IterativeMatcher: ...
    @matcher.setter
    def matcher(self, matcher: IterativeMatcherIn) -> None: ...
    def __init__(
        self,
        updater: Updater,
        matcher: IterativeMatcherIn,
    ) -> None: ...
    def update(
        self, circuit: Circuit, fancy_validate: Optional[bool] = None, assert_any_found: Optional[bool] = None
    ) -> Circuit: ...
    def __call__(
        self, circuit: Circuit, fancy_validate: Optional[bool] = None, assert_any_found: Optional[bool] = None
    ) -> Circuit:
        """Alias for update"""

class Getter:
    default_fancy_validate: bool
    """Fancy validation checks that all of the matchers matched something. For
    instance, each name/type/regex should match something. See tests for more details."""
    def __init__(self, default_fancy_validate: bool = False) -> None: ...
    def get(
        self, circuit: Circuit, matcher: IterativeMatcherIn, fancy_validate: Optional[bool] = None
    ) -> set[Circuit]: ...
    def __call__(
        self, circuit: Circuit, matcher: IterativeMatcherIn, fancy_validate: Optional[bool] = None
    ) -> set[Circuit]:
        """Alias for get"""
    def get_unique_op(
        self, circuit: Circuit, matcher: IterativeMatcherIn, fancy_validate: Optional[bool] = None
    ) -> Optional[Circuit]: ...
    def get_unique(
        self, circuit: Circuit, matcher: IterativeMatcherIn, fancy_validate: Optional[bool] = None
    ) -> Circuit: ...
    def get_paths(self, circuit: Circuit, matcher: IterativeMatcherIn) -> Dict[Circuit, Path]:
        """Return a dict where each matched node is associated with one arbitrary path to it."""
    def get_all_paths(self, circuit: Circuit, matcher: IterativeMatcherIn) -> Dict[Circuit, Paths]:
        """Return a dict where each matched node is associated with a list of every path to it."""
    def validate(self, circuit: Circuit, matcher: IterativeMatcherIn) -> None: ...
    def bind(self, matcher: IterativeMatcherIn) -> BoundGetter: ...

class BoundGetter:
    getter: Getter
    @property
    def matcher(self) -> IterativeMatcher: ...
    @matcher.setter
    def matcher(self, matcher: IterativeMatcherIn) -> None: ...
    def __init__(self, getter: Getter, matcher: IterativeMatcherIn) -> None: ...
    def get(self, circuit: Circuit, fancy_validate: Optional[bool] = None) -> set[Circuit]: ...
    def __call__(self, circuit: Circuit, fancy_validate: Optional[bool] = None) -> set[Circuit]:
        """Alias for get"""
    def get_unique_op(self, circuit: Circuit, fancy_validate: Optional[bool] = None) -> Optional[Circuit]: ...
    def get_unique(self, circuit: Circuit, fancy_validate: Optional[bool] = None) -> Circuit: ...
    def get_paths(self, circuit: Circuit) -> Dict[Circuit, Path]:
        """Return a dict where each matched node is associated with one arbitrary path to it."""
    def get_all_paths(self, circuit: Circuit) -> Dict[Circuit, Paths]:
        """Return a dict where each matched node is associated with a list of every path to it."""
    def validate(self, circuit: Circuit) -> None: ...

class AnyFound:
    def __init__(self) -> None: ...
    def are_any_found(self, circuit: Circuit, matcher: IterativeMatcherIn) -> bool: ...
    def __call__(self, circuit: Circuit, matcher: IterativeMatcherIn) -> bool:
        """Alias for are_any_found"""
    def bind(self, matcher: IterativeMatcherIn) -> BoundAnyFound: ...

class BoundAnyFound:
    any_found: AnyFound
    @property
    def matcher(self) -> IterativeMatcher: ...
    @matcher.setter
    def matcher(self, matcher: IterativeMatcherIn) -> None: ...
    def __init__(self, any_found: AnyFound, matcher: IterativeMatcherIn) -> None: ...
    def are_any_found(self, circuit: Circuit) -> bool: ...
    def __call__(self, circuit: Circuit) -> bool:
        """Alias for are_any_found"""

class Expander:
    """
    Run multiple updates in parallel while using expand_node to allow for adding batch dims or setting symbolic sizes.
    See `interp/demos/rust_circuit/modules_and_symbols.py` for examples.
    """

    ban_multiple_matches_on_node: bool
    default_fancy_validate: bool
    @property
    def replacements(self) -> List[Transform]: ...
    @property
    def matchers(self) -> List[IterativeMatcher]: ...
    @property
    def suffix(self) -> Optional[str]: ...
    def __init__(
        self,
        *expanders: Tuple[IterativeMatcherIn, TransformIn],
        ban_multiple_matches_on_node: bool = False,
        default_fancy_validate: bool = False,
        default_assert_any_found: bool = False,
        suffix: Optional[str] = None,
    ) -> None: ...
    def batch(
        self, circuit: Circuit, fancy_validate: Optional[bool] = None, assert_any_found: Optional[bool] = None
    ) -> Circuit: ...
    def __call__(
        self, circuit: Circuit, fancy_validate: Optional[bool] = None, assert_any_found: Optional[bool] = None
    ) -> Circuit:
        """Alias for batch"""

class TensorCacheRrfs(object):
    def __init__(self, cutoff: int, small_capacityint, large_capacityint, device: str) -> None: ...
    def get_tensor(self, prefix: str) -> torch.Tensor: ...
    def get_tensor_if_cached(self, prefix: str) -> Optional[torch.Tensor]: ...

def check_evaluable(circuit: Circuit) -> None: ...
def evaluate(circuit: Circuit) -> None: ...
def symbolic_sizes() -> List[int]: ...
def add_collapse_scalar_inputs(add: Add) -> Optional[Add]: ...
def add_deduplicate(add: Add) -> Optional[Add]: ...
def add_flatten_once(add: Add) -> Optional[Add]: ...
def remove_add_few_input(add: Add) -> Optional[Add]: ...
def add_pull_removable_axes(add: Add, remove_non_common_axes: bool) -> Optional[Circuit]: ...
def einsum_flatten_once(einsum: Einsum) -> Optional[Einsum]: ...
def einsum_elim_identity(einsum: Einsum) -> Optional[Circuit]: ...
def index_merge_scalar(index: Index) -> Optional[Circuit]: ...
def index_elim_identity(index: Index) -> Optional[Circuit]: ...
def index_fuse(index: Index) -> Optional[Index]: ...
def rearrange_fuse(node: Rearrange) -> Optional[Rearrange]: ...
def rearrange_merge_scalar(rearrange: Rearrange) -> Optional[Circuit]: ...
def rearrange_elim_identity(rearrange: Rearrange) -> Optional[Circuit]: ...
def concat_elim_identity(concat: Concat) -> Optional[Circuit]: ...
def concat_elim_split(concat: Concat) -> Optional[Index]: ...
def concat_merge_uniform(concat: Concat) -> Optional[Concat]: ...
def concat_pull_removable_axes(concat: Concat) -> Optional[Circuit]: ...
def generalfunction_pull_removable_axes(node: GeneralFunction) -> Optional[Circuit]: ...
def generalfunction_merge_inverses(node: GeneralFunction) -> Optional[Circuit]: ...
def generalfunction_special_case_simplification(node: GeneralFunction) -> Optional[Circuit]: ...
def generalfunction_evaluate_simple(node: GeneralFunction) -> Optional[Circuit]: ...
def generalfunction_gen_index_const_to_index(node: GeneralFunction) -> Optional[Circuit]: ...
def einsum_pull_removable_axes(node: Einsum) -> Optional[Circuit]: ...
def scatter_pull_removable_axes(node: Scatter) -> Optional[Circuit]: ...
def add_make_broadcasts_explicit(node: Add) -> Optional[Add]: ...
def make_broadcast(node: Circuit, out_shape: Shape) -> Optional[Circuit]: ...
def distribute_once(
    node: Einsum,
    operand_idx: int,
    do_broadcasts: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> Add: ...
def distribute_all(node: Einsum) -> Optional[Add]: ...
def einsum_of_permute_merge(node: Einsum) -> Optional[Einsum]: ...
def permute_of_einsum_merge(node: Rearrange) -> Optional[Einsum]: ...
def einsum_elim_zero(node: Einsum) -> Optional[Scalar]: ...
def einsum_merge_scalars(node: Einsum) -> Optional[Einsum]: ...
def push_down_index_once(
    node: Index, allow_partial_pushdown: bool = False, suffix: Optional[str] = None
) -> Circuit: ...
def deep_push_down_index_raw(node: Circuit, min_size: Optional[int] = None) -> Circuit: ...
def index_split_axes(node: Index, top_axes: Set[int]) -> Optional[Index]: ...
def add_elim_zeros(node: Add) -> Optional[Add]: ...
def strip_names_and_tags(node: Circuit) -> Circuit: ...
def deep_canonicalize(node: Circuit) -> Circuit: ...
def canonicalize_node(node: Circuit) -> Circuit: ...
def deep_normalize(node: Circuit) -> Circuit:
    """canon rearrange spec (without special casing 1s) + normalize einsum ints"""

def normalize_node(node: Circuit) -> Circuit:
    """canon rearrange spec (without special casing 1s) + normalize einsum ints"""

def deep_maybe_distribute(node: Circuit, settings: OptimizationSettings) -> Circuit: ...
def maybe_distribute(node: Einsum, settings: OptimizationSettings) -> Optional[Circuit]: ...
def einsum_nest_optimize(node: Einsum) -> Optional[Einsum]: ...
def deep_optimize_einsums(node: Circuit) -> Circuit: ...
def einsum_permute_to_rearrange(node: Einsum) -> Optional[Circuit]: ...
def einsum_nest_path(node: Einsum, path: List[List[int]]) -> Einsum: ...
def scatter_elim_identity(node: Scatter) -> Optional[Circuit]: ...
def einsum_pull_scatter(node: Einsum) -> Optional[Circuit]: ...
def add_pull_scatter(node: Add) -> Optional[Circuit]: ...
def index_einsum_to_scatter(node: Index) -> Optional[Circuit]: ...

class OptimizingSymbolicSizeWarning(UserWarning): ...

def optimize_circuit(circuit: Circuit, settings: OptimizationSettings = OptimizationSettings()) -> Circuit: ...
def cast_circuit(circuit: Circuit, device_dtype: TorchDeviceDtypeOp) -> Circuit: ...
def scatter_to_concat(scatter: Scatter) -> Circuit: ...
def count_nodes(circuit: Circuit) -> int: ...
def scheduled_evaluate(circuit: Circuit, settings: OptimizationSettings) -> torch.Tensor: ...
def optimize_and_evaluate(
    circuit: Circuit, settings: OptimizationSettings = OptimizationSettings()
) -> torch.Tensor: ...
def optimize_and_evaluate_many(
    circuits: List[Circuit], settings: OptimizationSettings = OptimizationSettings()
) -> List[torch.Tensor]: ...
def circuit_to_schedule(circuit: Circuit, settings: OptimizationSettings) -> Schedule: ...
def circuit_to_schedule_many(circuits: List[Circuit], settings: OptimizationSettings) -> Schedule: ...
def optimize_to_schedule(circuit: Circuit, settings: OptimizationSettings = OptimizationSettings()) -> Schedule: ...
def optimize_to_schedule_many(
    circuits: List[Circuit], settings: OptimizationSettings = OptimizationSettings()
) -> Schedule: ...
def flat_concat(circuits: List[Circuit]) -> Concat: ...
def flat_concat_back(circuits: List[Circuit]) -> Tuple[Concat, List[Circuit]]: ...
def deep_heuristic_nest_adds(circuit: Circuit) -> Circuit: ...
def generalfunction_pull_concat(circuit: GeneralFunction) -> Optional[Concat]: ...
def concat_fuse(circuit: Concat) -> Optional[Concat]: ...
def concat_drop_size_zero(circuit: Concat) -> Optional[Concat]: ...
def index_concat_drop_unreached(circuit: Index) -> Optional[Index]: ...
def einsum_pull_concat(circuit: Einsum) -> Optional[Circuit]: ...
def add_pull_concat(circuit: Add) -> Optional[Circuit]: ...
def split_to_concat(circuit: Circuit, axis: int, sections: List[int]) -> Optional[Circuit]:
    """sections is a list of the desired widths"""

def deep_pull_concat_messy(circuit: Circuit, min_size: Optional[int]) -> Circuit: ...
def deep_pull_concat(circuit: Circuit, min_size: Optional[int] = None) -> Circuit: ...
def batch_to_concat(
    circuit: Circuit, axis: int, batch_size: int, allow_partial_batch: bool = True, keep_non_axis_leaves: bool = False
) -> Circuit: ...
def batch_einsum(einsum: Einsum, settings: OptimizationSettings) -> Circuit: ...
def set_named_axes(circuit: Circuit, named_axes: Dict[int, str]) -> Circuit: ...
def propagate_named_axes(circuit: Circuit, named_axes: Dict[int, str], do_merge: bool) -> Circuit: ...
def toposort_circuit(circuit: Circuit) -> List[Circuit]: ...
def einsum_push_down_trace(circuit: Einsum) -> Optional[Einsum]: ...
def einsum_concat_to_add(circuit: Einsum) -> Optional[Einsum]: ...
def add_pull_diags(circuit: Add) -> Optional[Circuit]: ...
def concat_repeat_to_rearrange(circuit: Concat) -> Optional[Concat]: ...
def add_outer_product_broadcasts_on_top(circuit: Add) -> Optional[Add]: ...
def extract_add(add: Add, sub: Add) -> Optional[Add]: ...
def add_fuse_scalar_multiples(add: Add) -> Optional[Add]: ...
def concat_to_scatter(concat: Concat) -> Optional[Scatter]: ...
def inline_single_callsite_modules(circuit: Circuit) -> Circuit: ...
def replace_all_randn_seeded(circuit: Circuit) -> Circuit: ...
def opt_eval_each_subcircuit_until_fail(circuit: Circuit, settings: OptimizationSettings) -> None: ...
def compute_self_hash(circuit: Circuit) -> bytes: ...
def diff_circuits(
    new: Circuit,
    old: Circuit,
    options: PrintOptions = PrintOptions(),
    require_child_count_same: bool = True,
    require_child_shapes_same: bool = False,
    require_name_same: bool = True,
    print_legend: bool = True,
    same_self_color: CliColor = "Blue",
    same_color: CliColor = None,
    new_color: CliColor = "Green",
    removed_color: CliColor = "Red",
) -> str: ...

# circuit manipulation functions
def deep_map(circuit: Circuit, fn: Callable[[Circuit], Circuit]) -> Circuit: ...
def deep_map_preorder(circuit: Circuit, fn: Callable[[Circuit], Circuit]) -> Circuit: ...
def filter_nodes(circuit: Circuit, fn: Callable[[Circuit], bool]) -> List[Circuit]: ...
def replace_nodes(circuit: Circuit, map: Dict[Circuit, Circuit]) -> Circuit: ...
def path_get(circuit: Circuit, path: List[int]) -> Circuit: ...
def einsum_elim_removable_axes_weak(circuit: Einsum) -> Optional[Circuit]: ...
def add_elim_removable_axes_weak(circuit: Add) -> Optional[Circuit]: ...
def update_nodes(
    circuit: Circuit, matcher: Callable[[Circuit], bool], mapper: Callable[[Circuit], Circuit]
) -> Circuit: ...
def update_path(circuit: Circuit, path: List[int], updater: Callable[[Circuit], Circuit]) -> Circuit: ...
def expand_node(circuit: Circuit, inputs: List[Circuit]) -> Circuit:
    """
    Typically users should use Expander.

    Expand the shape of single node by replacing the inputs with new inputs with (possibly) expanded shapes.

    This supports adding batch dims or editing the shapes.
    This is effectively like applying Expander to replace all children of the root circuit.

    More precise spec:
    For each child, the additional dimensions are considered batch dimensions.
    That is, if we have a child with shape:
    [      a, b, c]
    And it's replaced with a circuit of shape:
    [d, e, a, b, c]
    d and e are considered batch dims.
    The batch dims for all inputs are 'right aligned' (and must be equal at all lined up sizes).
    We batch over these dimensions (computationally equivalent to running the
    circuit repeatedly for each value). This batching always adds additional dims to
    the circuit with the same shape as the batch dims.

    If already existing dimensions are changed, we just replace the circuit
    with these new dimensions (and potentially error). That is, except for
    symbolic sizes which are constrained such that the expansion is possible.


    So if we have the following shape changes for our children:
    [          a, c, d] -> [     b_1, A, C, D]
    [             e, f] -> [     b_1,    E, F]
    [                g] -> [b_0, b_1,       G]

    Then, we'll end up with an output shape of [b_0, b_1, *] where the exact shape
    of '*' will depend on how the modified (expanded) sizes: A, C, ..., G play out.
    """

def save_tensor_rrfs(tensor: torch.Tensor) -> str: ...  # string is base16 key
def tensor_from_hash(hash: str) -> torch.Tensor: ...
def substitute_all_modules(circuit: Circuit) -> Circuit: ...
def conform_all_modules(circuit: Circuit) -> Circuit:
    """
    Can be used to reshape symbols bound in modules to the shape of their inputs. This both adds batch dimensions and sets symbolic sizes. This makes all module specs have the 'real' sizes everywhere.
    """

def get_children_with_symbolic_sizes(circuit: Circuit) -> Set[Circuit]: ...
def any_children_with_symbolic_sizes(circuit: Circuit) -> bool: ...
def replace_expand_bottom_up(circuit: Circuit, f: Callable[[Circuit], Optional[Circuit]]) -> Circuit: ...
def replace_expand_bottom_up_dict(circuit: Circuit, dict: Dict[Circuit, Circuit]) -> Circuit: ...
def prefix_all_names(circuit: Circuit, prefix: str) -> Circuit: ...
def extract_rewrite(
    circuit: Circuit,
    input_matcher: Matcher,
    prefix_to_strip: Optional[str] = None,
    module_name: Optional[str] = None,
    require_all_args=True,
    check_unique_arg_names=True,
    # defaults to making symbol with just_name_shape (batchable=True,expandable=True,ban_non_symbolic_size_expand=False)
    circuit_to_arg_spec: Optional[Callable[[Circuit], ModuleArgSpec]] = None,
) -> Module: ...

class BindItem:
    """
    Utility class for module_new_bind. See ModuleArgSpec docs.

    matcher should match exactly one circuit.
    """

    @property
    def matcher(self) -> IterativeMatcher: ...
    @matcher.setter
    def matcher(self, matcher: IterativeMatcherIn) -> None: ...
    input_circuit: Circuit
    batchable: bool
    expandable: bool
    ban_non_symbolic_size_expand: bool
    def __init__(
        self,
        matcher: IterativeMatcherIn,
        input_circuit: Circuit,
        batchable: bool = True,
        expandable: bool = True,
        ban_non_symbolic_size_expand: bool = True,
    ) -> None: ...

Binder = Union[Tuple[IterativeMatcherIn, Circuit], BindItem]

def module_new_bind(
    spec_circuit: Circuit, *binders: Binder, check_unique_arg_names: bool = True, name: Optional[str] = None
) -> Module:
    """
    Create a ModuleSpec using the circuits matched by the binders as input symbols
    and feed the circuit provided by the binders as input to the module.

    A binder of the form (it_matcher, circ) behaves like BindItem(it_matcher, circ).
    """

NestedModuleNamer = Callable[[Circuit, Circuit, List[Module], Optional[int]], Optional[str]]
"""
Args: base_circuit, running_circuit, modules, pushed_overall_mod_count

modules are inner to outer.

pushed_overall_mod_count is the overall number of modules which ended up pushed
to the node we're terminating at. If we're not terminating, None.

Generally you should use base_circuit and modules and ignore running_circuit.

In most cases, base_circuit == running_circuit, but not if you push down
modules with flatten_modules=False and have multiple nested modules.
In the nested module case, the name of running_circuit will depend on previous
calls to NestedModuleNamer.
"""

def default_nested_module_namer(bind_name: str = "bind") -> NestedModuleNamer: ...

ModuleConstructCallback = Callable[[Module, List[Module], int], Circuit]
"""
Args: circuit, modules, pushed_overall_mod_count

modules are inner to outer.

pushed_overall_mod_count is the overall number of modules which ended up pushed
to the node we're terminating at.
"""

class ModulePusher:
    @property
    def flatten_modules(self) -> bool: ...
    @property
    def module_construct_callback(self) -> ModuleConstructCallback: ...
    @property
    def bind_encountered_symbols(self) -> bool: ...
    @property
    def namer(self) -> NestedModuleNamer: ...
    has_sym: AnyFound
    def __init__(
        self,
        flatten_modules: bool = True,
        module_construct_callback: ModuleConstructCallback = ModulePusher.remove_unused_callback(
            add_suffix_on_remove_unused=False, elim_no_input_modules=True
        ),
        bind_encountered_symbols: bool = True,
        namer: NestedModuleNamer = default_nested_module_namer(),
    ) -> None: ...
    def __call__(
        self, circuit: Circuit, traversal: IterativeMatcherIn, skip_module: IterativeMatcherIn = IterativeMatcher(False)
    ) -> Circuit:
        """alias for push_down_modules"""
    def push_down_modules(
        self, circuit: Circuit, traversal: IterativeMatcherIn, skip_module: IterativeMatcherIn = IterativeMatcher(False)
    ) -> Circuit: ...
    def get_push_down_modules(
        self, circuit: Circuit, get: IterativeMatcherIn, skip_module: IterativeMatcherIn = IterativeMatcher(False)
    ) -> Set[Circuit]: ...
    def get_unique_op_push_down_modules(
        self, circuit: Circuit, get: IterativeMatcherIn, skip_module: IterativeMatcherIn = IterativeMatcher(False)
    ) -> Optional[Circuit]: ...
    def get_unique_push_down_modules(
        self, circuit: Circuit, get: IterativeMatcherIn, skip_module: IterativeMatcherIn = IterativeMatcher(False)
    ) -> Circuit: ...
    @staticmethod
    def remove_and_elim_callback(
        remove_unused_inputs: bool = True, add_suffix_on_remove_unused: bool = False, elim_no_input_modules: bool = True
    ) -> ModuleConstructCallback:
        """helper for configuring between below callbacks"""
    @staticmethod
    def remove_unused_callback(
        add_suffix_on_remove_unused: bool = False, elim_no_input_modules: bool = True
    ) -> ModuleConstructCallback: ...
    @staticmethod
    def noop_callback() -> ModuleConstructCallback: ...
    @staticmethod
    def elim_no_input_modules_callback() -> ModuleConstructCallback: ...

MaybeUpdate = Callable[[Circuit], Optional[Circuit]]

def default_update_bindings_nested_namer(
    bind_name: str = "upd_bind", short_if_not_leaf: bool = True, keep_name_if_not_leaf: bool = False
) -> NestedModuleNamer: ...
def update_bindings_nested(
    circuit: Circuit,
    update: MaybeUpdate,
    matcher: IterativeMatcherIn,
    namer: NestedModuleNamer = default_update_bindings_nested_namer(),
    skip_module: IterativeMatcherIn = IterativeMatcher(False),
    run_update_on_new_spec_circuits: bool = False,
    flatten_modules: bool = False,
) -> Circuit:
    """
    BE WARNED: this function is virtually untested!

    Also note that namer is used somewhat differently, TODO: doc
    """

def extract_symbols(
    circuit: Circuit,
    symbols: Set[Symbol],
    use_elim_no_input_modules: bool = True,
    conform_batch_if_needed: bool = False,
    traversal: IterativeMatcherIn = new_traversal(),
) -> Module: ...
def extract_symbols_get(
    circuit: Circuit,
    get: IterativeMatcherIn,
    use_elim_no_input_modules: bool = True,
    conform_batch_if_needed: bool = False,
    traversal: IterativeMatcherIn = new_traversal(),
) -> Module: ...
def fuse_concat_modules(
    circuit: Circuit,
    modules: List[Module],
    name: Optional[str] = None,
) -> Circuit: ...
def apply_in_traversal(circuit: Circuit, traversal: IterativeMatcherIn, f: Callable[[Circuit], Circuit]) -> Circuit: ...
def replace_outside_traversal_symbols(
    circuit: Circuit, traversal: IterativeMatcherIn, namer: Optional[Callable[[Circuit], Optional[str]]] = None
) -> Tuple[Circuit, Dict[Circuit, Circuit]]: ...
def elim_empty_module(module: Module) -> Optional[Circuit]: ...
def elim_no_input_module(module: Module) -> Optional[Circuit]: ...
def module_remove_unused_inputs(
    m: Module, add_suffix_on_remove_unused: bool = True, use_elim_no_input_module: bool = True
) -> Circuit: ...
def deep_module_remove_unused_inputs(
    circuit: Circuit,
    add_suffix_on_remove_unused: bool = True,
    use_elim_no_input_module: bool = True,
    elim_empty: bool = False,
) -> Circuit: ...
def extract_rewrite_raw(
    circuit: Circuit,
    input_specs: List[Tuple[Circuit, ModuleArgSpec]],
    prefix_to_strip: Optional[str] = None,
    module_name: Optional[str] = None,
    require_all_args=True,
    check_unique_arg_names=True,
) -> Module: ...
def circuit_is_leaf(circuit: Circuit) -> bool: ...
def circuit_is_irreducible_node(circuit: Circuit) -> bool: ...
def circuit_is_leaf_constant(circuit: Circuit) -> bool: ...
def circuit_is_var(circuit: Circuit) -> bool: ...
def circuit_server_serve(url: str, device: str, tensor_cache: Optional[TensorCacheRrfs]): ...
def visit_circuit(circuit: Circuit, f: Callable[[Circuit], None]) -> None: ...
def all_children(circuit: Circuit) -> Set[Circuit]: ...
def print_circuit_type_check(x: Type[Circuit]) -> Type[Circuit]: ...
def hash_tensor(x: torch.Tensor) -> bytes: ...
def default_var_matcher() -> IterativeMatcher: ...
def save_tensor(tensor: torch.Tensor, force=False): ...
def sync_all_unsynced_tensors(parallelism: int = 15): ...
def sync_specific_tensors(tensor_hash_prefixes: list[str], parallelism: int = 15): ...
def get_tensor_prefix(prefix: str) -> torch.Tensor: ...
def migrate_tensors_from_old_dir(dir: str): ...

# simplification
def compiler_simp(circuit: Circuit) -> Circuit: ...
def compiler_simp_step(circuit: Circuit) -> Optional[Circuit]: ...
def simp(circuit: Circuit) -> Circuit: ...
def default_hash_seeder(base_seed: Optional[int] = None) -> Callable[[Circuit], int]:
    """
    Get a seeder which return a seed for probs and group based on the hash.
    The base_seed is xor'ed in.

    If you pass 'None', this will generate a new base_seed using randomness
    from torch (so you can seed torch to make the reproducible)
    """

class RandomSampleSpec(SampleSpec):
    probs_and_group_evaluation_settings: OptimizationSettings
    seeder: Callable[[Circuit], int]
    def __init__(
        self,
        shape: Shape = (),
        replace: bool = True,
        simplify: bool = True,
        probs_and_group_evaluation_settings: OptimizationSettings = OptimizationSettings(),
        seeder: Optional[Callable[[Circuit], int]] = None,
    ):
        """
        If seeder is None, initialize a new random seeder using default_hash_seeder()

        seeder takes in probs and group and returns seed.

        simplify: should we compute the resulting Multinomial & simplify the resulting gen_index into a normal Index?

        NOTE: the seeder should be deterministic, otherwise, you'll end up with
        different seeds for the same probs and group! (see default_hash_seeder for instance)
        """
    @property
    def shape(self) -> Shape: ...
    @property
    def replace(self) -> bool: ...
    @property
    def simplify(self) -> bool: ...

class RunDiscreteVarAllSpec:
    def __init__(
        self,
        groups: Sequence[Circuit],
        subsets: Optional[Sequence[slice]] = None,
    ): ...
    @property
    def groups(self) -> List[Circuit]: ...
    @property
    def subsets(self) -> List[slice]: ...
    @staticmethod
    def create_full_from_circuits(*circuits: Circuit) -> RunDiscreteVarAllSpec: ...

SampleSpecIn = RandomSampleSpec | RunDiscreteVarAllSpec

class Sampler:
    def __init__(
        self,
        sample_spec: SampleSpecIn,
        var_matcher: IterativeMatcherIn = default_var_matcher(),
        cumulant_matcher: IterativeMatcherIn = Cumulant,
        suffix: Optional[str] = "sample",
        run_on_sampled: TransformIn = Transform.ident(),
    ):
        """
        See docs for `estimate` for details on `cumulant_matcher`.

        run_on_sampled is useful for batching over samples (via batch_to_concat)
        """
    @property
    def expander(self) -> Expander: ...
    @property
    def cumulant_matcher(self) -> IterativeMatcher: ...
    @property
    def sample_spec(self) -> SampleSpec: ...
    @property
    def run_on_sampled(self) -> Transform: ...
    def estimate(self, circuit: Circuit) -> Circuit:
        """
        Recursively estimate all cumulants matched by `cumulant_matcher`
        This doesn't require that the circuit is constant.
        If your matcher allows for recursive matching, then it will estimate nested cumulants.
        The default matcher estimates all nested cumulants.

        Note the the cumulant estimation process involves creating nested
        cumulants and then recusively estimating these, so the traversal will
        hit cumulants which weren't originally in the circuit - keep this in mind when passing
        a `cumulant_matcher`.
        """
    def sample(self, circuit: Circuit) -> Circuit:
        """
        Runs sample on circuit. See docs for get_sampler on SampleSpec.
        Also expands to apppropriate shape (even if there aren't any vars) and calls run_on_sampled.
        """
    def estimate_and_sample(self, circuit: Circuit) -> Circuit:
        """
        first run estimate, then run sample
        """
    def __call__(self, c: Circuit) -> Circuit:
        """
        alias for estimate_and_sample
        """

def factored_cumulant_expectation_rewrite(circuit: Cumulant) -> Circuit: ...

IntOrMatcher = Union[int, IterativeMatcherIn]

class NestRest:
    """See ``interp/demos/nest.py``"""

    flat: bool
    def __init__(self, flat: bool = False) -> None: ...

class NestMatcher:
    """See ``interp/demos/nest.py``"""

    @property
    def matcher(self) -> IterativeMatcher: ...
    @matcher.setter
    def matcher(self, matcher: IterativeMatcherIn) -> None: ...
    flat: bool
    assert_exists: bool
    assert_unique: bool
    fancy_validate: bool
    def __init__(
        self,
        matcher: IterativeMatcherIn,
        flat: bool = False,
        assert_exists: bool = True,
        assert_unique: bool = False,
        fancy_validate: bool = False,
    ) -> None: ...

NestEinsumsSpecMultiple = Union[NestRest, NestMatcher, Sequence[NestEinsumsSpec]]  # type: ignore
NestEinsumsSpecSub = Union[IntOrMatcher, NestEinsumsSpecMultiple]  # type: ignore
NestAddsSpecMultiple = Union[NestRest, NestMatcher, Sequence[NestAddsSpec]]  # type: ignore
NestAddsSpecSub = Union[IntOrMatcher, NestAddsSpecMultiple]  # type: ignore

class NestEinsumsSpecInfo:
    """
    ..note:: see ``interp/demos/nest.py`` for an interactive demonstration of how this works

    name: new name for einsum formed by spec
    out_axes_perm: permutation to apply to the outaxes for this einsum
    shrink_out_axes: see demo
    """

    spec: NestEinsumsSpecMultiple
    name: Optional[str]
    out_axes_perm: Optional[Sequence[int]]
    shrink_out_axes: bool
    def __init__(
        self,
        spec: NestEinsumsSpecMultiple,
        name: Optional[str] = None,
        out_axes_perm: Optional[Sequence[int]] = None,
        shrink_out_axes: bool = False,
    ) -> None: ...

class NestAddsSpecInfo:
    """
    ..note:: see ``interp/demos/nest.py`` for an interactive demonstration of how this works

    name: new name for einsum formed by spec
    """

    spec: NestAddsSpecMultiple
    name: Optional[str]
    def __init__(
        self,
        spec: NestAddsSpecMultiple,
        name: Optional[str] = None,
    ) -> None: ...

NestEinsumsSpec = Union[NestEinsumsSpecSub, NestEinsumsSpecInfo]  # type: ignore
NestAddsSpec = Union[NestAddsSpecSub, NestAddsSpecInfo]  # type: ignore

def einsum_flatten(einsum: Einsum, traversal: IterativeMatcherIn = new_traversal()) -> Einsum: ...
def add_flatten(add: Add, traversal: IterativeMatcherIn = new_traversal()) -> Add: ...
def nest_einsums(einsum: Einsum, spec: NestEinsumsSpecSub, traversal: IterativeMatcherIn = new_traversal()) -> Einsum:
    """
    Rearranges the order and nesting of einsums in a nested einsum.

    ..note:: see ``interp/demos/nest.py`` for an interactive demonstration of how this works

    ``traversal`` allows for only matching and rearrange a subset of the
    einsums - whatever the traversal traverses to (this isn't what it matches,
    just what nodes it traverses before terminating.) See demo for examples
    """

def nest_adds(add: Add, spec: NestAddsSpecSub, traversal: IterativeMatcherIn = new_traversal()) -> Add:
    """
    Rearranges the order and nesting of adds in a nested add.

    ..note:: see ``interp/demos/nest.py`` for an interactive demonstration of how this works

    ``traversal`` allows for only matching and rearrange a subset of the
    adds - whatever the traversal traverses to (this isn't what it matches,
    just what nodes it traverses before terminating.) See demo for examples
    """

def default_index_traversal() -> IterativeMatcher:
    """
    ``traversal.filter(term_early_at=[Index, Array])``
    """

def push_down_index(
    index: Index,
    traversal: IterativeMatcherIn = default_index_traversal(),
    suffix: Optional[str] = None,
    allow_partial_pushdown: bool = False,
    elim_identity: bool = True,
) -> Circuit: ...
def traverse_until_depth(depth: Optional[int] = None) -> IterativeMatcher:
    """
    Traverse until specified depth (inclusive), does not match.

    None is infinite depth.
    """

def distribute(
    einsum: Einsum,
    operand_idx: int,
    traversal: IterativeMatcherIn = traverse_until_depth(1),
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    allow_partial_distribute: bool = True,  # noop when fail to distribute at one point
    do_broadcasts: bool = True,
) -> Circuit:
    """
    Distribute recursively.

    If allow_partial_distribute = False, all summands traversed should be Add nodes.
    For example, a*((b+c)+(d+e)) can be distributed at depth 2 allow_partial_distribute = False
    but not a*((b+c)+d)"""

def rewrite_cum_to_circuit_of_cum(
    cumulant: Cumulant,
    node: Circuit,
    namer: Optional[Callable[[Cumulant], Optional[str]]] = None,
    on_sub_cumulant_fn: Optional[Callable[[Cumulant], Circuit]] = None,
):
    """
    Express a cumulant of circuits as a circuit of cumulants by expanding the node matching `node`.

    Returns 1 on empty cumulants (even if no node is matched).
    """

def kappa_term(
    args: Sequence[Sequence[Tuple[int, Circuit]]],
    namer: Optional[Callable[[Cumulant], Optional[str]]] = None,
    on_sub_cumulant_fn: Optional[Callable[[Cumulant], Circuit]] = None,
) -> Tuple[Einsum, Sequence[Cumulant]]: ...

class CharTokenizer(object):
    """rudimentary tokenizer, made for paren balancer. each token must be 1 character and 1 byte."""

    start: int
    end: int
    pad: int
    pad_width: int
    mapping: Dict[int, int]
    error_if_over: bool
    def __init__(
        self, start: int, end: int, pad: int, pad_width: int, mapping: Dict[str, int], error_if_over: bool
    ) -> None: ...
    def tokenize_strings(self, strings: List[str]) -> torch.Tensor: ...

def oom_fmt(x: int) -> str: ...
