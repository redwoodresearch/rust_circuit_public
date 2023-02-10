from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Union, TypeVar, Sequence, Literal

from dataclasses import dataclass

import rust_circuit.index_util.internal.parse as parse


@dataclass(frozen=True, eq=True)
class OutAxisIdx:
    axis: int


@dataclass(frozen=True, eq=True)
class IntLitIdx:
    value: int


@dataclass(frozen=True, eq=True)
class PosDataIdx:
    pass


if TYPE_CHECKING:
    T = TypeVar("T")


def is_unique(lst: list[T]) -> bool:
    return len(set(lst)) == len(lst)


OUT_NAME: str = "out"

if TYPE_CHECKING:
    IdxExpr = Union[OutAxisIdx, IntLitIdx, PosDataIdx]


@dataclass
class IdxRenderer:
    axis_names: list[str]
    pos_name: str
    pos_indices: list[IdxExpr]

    def render_indices(self, indices: list[IdxExpr], out: list[str]) -> None:
        if len(indices) > 0:
            out.append("[")
            for i, idx in enumerate(indices):
                if i > 0:
                    out.append(", ")
                self.render_idx(idx, out)
            out.append("]")

    def render_idx(self, idx: IdxExpr, out: list[str]) -> None:
        if isinstance(idx, OutAxisIdx):
            out.append(self.axis_names[idx.axis])
        elif isinstance(idx, IntLitIdx):
            out.append(str(idx.value))
        elif isinstance(idx, PosDataIdx):
            out.append(self.pos_name)
            self.render_indices(self.pos_indices, out)
        else:
            assert False


@dataclass(repr=False)
class GatherSpec:
    out_rank: int
    src_indices: list[IdxExpr]
    pos_indices: list[IdxExpr]

    axis_names: list[str]
    src_name: str
    pos_name: str

    def __post_init__(self) -> None:
        assert sum(1 for i in self.src_indices if isinstance(i, PosDataIdx)) == 1
        assert all(isinstance(i, OutAxisIdx) or isinstance(i, IntLitIdx) for i in self.pos_indices)
        assert all(i.axis < self.out_rank for i in self.src_indices + self.pos_indices if isinstance(i, OutAxisIdx))
        assert all(parse.is_ident(n) for n in self.axis_names)
        assert len(self.axis_names) == self.out_rank
        assert parse.is_ident(self.src_name)
        assert parse.is_ident(self.pos_name)
        assert is_unique(self.axis_names + [self.src_name, self.pos_name, OUT_NAME])

    def render(self, out: list[str]) -> None:
        renderer = IdxRenderer(axis_names=self.axis_names, pos_name=self.pos_name, pos_indices=self.pos_indices)
        out.append(f"{OUT_NAME}")
        renderer.render_indices([OutAxisIdx(i) for i in range(self.out_rank)], out)
        out.append(" = ")
        out.append(self.src_name)
        renderer.render_indices(self.src_indices, out)

    def __str__(self) -> str:
        out: list[str] = []
        self.render(out)
        return "".join(out)

    def __repr__(self) -> str:
        return f"GatherSpec.parse({str(self)!r}, {[self.src_name, self.pos_name]!r})"

    @staticmethod
    def resolve(equation: parse.Equation, arr_names: list[str]) -> "GatherSpec":
        if equation.kind != "=":
            raise ValueError(f"Gather formula must be written with '=' operator; got: {equation.kind!r}")

        if len(arr_names) != 2:
            raise ValueError(f"Expected exactly two input array names for gather; got: {arr_names!r}")

        lhs = equation.lhs
        if not (isinstance(lhs, parse.IdentExpr) and lhs.name == OUT_NAME):
            raise ValueError(f"Invalid left hand side; expected {OUT_NAME!r} array, got: {lhs}")

        if not is_unique([OUT_NAME] + arr_names):
            raise ValueError("Array names must be unique")

        forbidden_axis_names = set([OUT_NAME] + arr_names)
        axis_names = []
        for i, e in enumerate(lhs.idx):
            if not (isinstance(e, parse.IdentExpr) and e.idx == []):
                raise ValueError(f"Invalid index name: {e}")
            if e.name in forbidden_axis_names:
                raise ValueError(f"Index name conflicts with a name already in scope: {e.name!r}")
            forbidden_axis_names.add(e.name)
            axis_names.append(e.name)
        axis_name_indices = {n: i for i, n in enumerate(axis_names)}

        out_rank = len(axis_names)

        rhs = equation.rhs
        if not isinstance(rhs, parse.IdentExpr):
            raise ValueError(f"Invalid right hand side; expected array access expression, got: {rhs}")
        src_name = rhs.name
        if src_name not in arr_names:
            raise ValueError(f"Invalid 'src' array name; expected one of {arr_names!r}, got: {src_name!r}")

        pos_name = [n for n in arr_names if n != src_name][0]

        def resolve_direct_index(e: parse.Expr) -> Union[OutAxisIdx, IntLitIdx]:
            if isinstance(e, parse.IdentExpr) and e.idx == []:
                axis_idx = axis_name_indices.get(e.name)
                if axis_idx is None:
                    raise ValueError(f"Index name not in scope; expected one of {axis_names!r}, got: {e.name!r}")
                return OutAxisIdx(axis_idx)
            elif isinstance(e, parse.IntExpr):
                return IntLitIdx(e.value)
            else:
                raise ValueError(f"Invalid index expression: {e}")

        src_indices: list[IdxExpr] = []
        pos_indices: Optional[list[IdxExpr]] = None
        for i, e in enumerate(rhs.idx):
            if isinstance(e, parse.IdentExpr) and e.name == pos_name:
                src_indices.append(PosDataIdx())
                if pos_indices is not None:
                    raise ValueError(f"Index array {pos_name!r} appears more than once")
                pos_indices = [resolve_direct_index(e2) for e2 in e.idx]
            else:
                src_indices.append(resolve_direct_index(e))

        if pos_indices is None:
            raise ValueError(f"Index array {pos_name!r} must be used")

        return GatherSpec(
            out_rank=out_rank,
            axis_names=axis_names,
            src_name=src_name,
            src_indices=src_indices,
            pos_name=pos_name,
            pos_indices=pos_indices,
        )

    @staticmethod
    def parse(s: str, arr_names: Sequence[str]) -> "GatherSpec":
        return GatherSpec.resolve(parse.Equation.parse(s), list(arr_names))


if TYPE_CHECKING:
    ScatterReduce = Optional[Literal["add", "multiply"]]


@dataclass
class ScatterSpec:
    dst_indices: list[IdxExpr]
    src_indices: list[IdxExpr]
    pos_indices: list[IdxExpr]

    axis_names: list[str]
    dst_name: str
    src_name: str
    pos_name: str

    reduce: ScatterReduce

    def __post_init__(self) -> None:
        assert sum(1 for i in self.dst_indices if isinstance(i, PosDataIdx)) == 1
        assert all(
            isinstance(i, OutAxisIdx) or isinstance(i, IntLitIdx)
            for indices in [self.src_indices, self.pos_indices]
            for i in indices
        )
        assert all(
            i.axis < len(self.axis_names)
            for indices in [self.dst_indices, self.src_indices, self.pos_indices]
            for i in indices
            if isinstance(i, OutAxisIdx)
        )
        assert all(parse.is_ident(n) for n in self.axis_names)
        assert parse.is_ident(self.dst_name)
        assert parse.is_ident(self.src_name)
        assert parse.is_ident(self.pos_name)
        assert is_unique(self.axis_names + [self.dst_name, self.src_name, self.pos_name])

    def render(self, out: list[str]) -> None:
        renderer = IdxRenderer(axis_names=self.axis_names, pos_name=self.pos_name, pos_indices=self.pos_indices)
        out.append(self.dst_name)
        renderer.render_indices(self.dst_indices, out)
        out.append(" <- ")
        out.append(self.src_name)
        renderer.render_indices(self.src_indices, out)

    def __str__(self) -> str:
        out: list[str] = []
        self.render(out)
        return "".join(out)

    def __repr__(self) -> str:
        return f"ScatterSpec({self!s}, {[self.dst_name, self.src_name, self.pos_name]!r})"

    @staticmethod
    def resolve(equation: parse.Equation, arr_names: list[str], reduce: ScatterReduce) -> "ScatterSpec":
        if equation.kind != "<-":
            raise ValueError(f"Scatter formula must be written with '<-' operator; got: {equation.kind!r}")

        if len(arr_names) != 3:
            raise ValueError(f"Expected exactly three input array names for scatter; got: {arr_names!r}")

        if not is_unique(arr_names):
            raise ValueError(f"Input array names must be unique; got: {arr_names!r}")

        lhs = equation.lhs
        if not isinstance(lhs, parse.IdentExpr):
            raise ValueError(f"Invalid left hand side; expected array access expression, got: {lhs}")
        dst_name = lhs.name
        if dst_name not in arr_names:
            raise ValueError(f"Invalid 'dst' array name; expected one of {arr_names!r}, got: {dst_name!r}")

        rhs = equation.rhs
        if not isinstance(rhs, parse.IdentExpr):
            raise ValueError(f"Invalid right hand side; expected array access expression, got: {rhs}")
        src_name = rhs.name
        if src_name not in arr_names:
            raise ValueError(f"Invalid 'src' array name; expected one of {arr_names!r}, got: {src_name!r}")

        pos_name = next(n for n in arr_names if n not in [dst_name, src_name])

        axis_names: list[str] = []
        axis_ids: dict[str, int] = {}

        def resolve_direct_index(err_ctx: str, e: parse.Expr) -> Union[OutAxisIdx, IntLitIdx]:
            if isinstance(e, parse.IdentExpr) and e.idx == []:
                if e.name == pos_name:
                    raise ValueError(f"Scatter index array {pos_name!r} cannot appear in indices for {err_ctx}")
                axis_idx = axis_ids.get(e.name)
                if axis_idx is None:
                    axis_idx = len(axis_names)
                    axis_names.append(e.name)
                    axis_ids[e.name] = axis_idx
                return OutAxisIdx(axis_idx)
            elif isinstance(e, parse.IntExpr):
                return IntLitIdx(e.value)
            else:
                raise ValueError(f"Invalid index expression in {err_ctx}: {e}")

        dst_indices: list[IdxExpr] = []
        pos_indices: Optional[list[IdxExpr]] = None
        for i, e in enumerate(lhs.idx):
            if isinstance(e, parse.IdentExpr) and e.name == pos_name:
                dst_indices.append(PosDataIdx())
                if pos_indices is not None:
                    raise ValueError(f"Scatter index array {pos_name!r} must be used exactly once")
                pos_indices = [resolve_direct_index(repr(pos_name), e2) for e2 in e.idx]
            else:
                dst_indices.append(resolve_direct_index(repr(dst_name), e))

        if pos_indices is None:
            raise ValueError(f"Scatter index array {pos_name!r} must be used")

        src_indices: list[IdxExpr] = [resolve_direct_index(repr(src_name), e) for e in rhs.idx]

        all_axes = set(range(len(axis_names)))
        dst_axes = set(i.axis for i in dst_indices if isinstance(i, OutAxisIdx))
        pos_axes = set(i.axis for i in pos_indices if isinstance(i, OutAxisIdx))

        missing_dst = all_axes - dst_axes
        if len(missing_dst) > 1:
            raise ValueError(
                f"At most one index variable can be missing from indices used to directly access {dst_name!r} array; "
                + f"found {len(missing_dst)} missing indices: {[axis_names[i] for i in missing_dst]!r}"
            )

        missing_lhs = all_axes - dst_axes - pos_axes
        if len(missing_lhs) > 0:
            raise ValueError(
                f"All index variables must be used on left hand side of scatter formula; "
                + f"found {len(missing_lhs)} missing indices: {[axis_names[i] for i in missing_lhs]!r}"
            )

        return ScatterSpec(
            dst_indices=dst_indices,
            src_indices=src_indices,
            pos_indices=pos_indices,
            axis_names=axis_names,
            dst_name=dst_name,
            src_name=src_name,
            pos_name=pos_name,
            reduce=reduce,
        )

    @staticmethod
    def parse(s: str, arr_names: Sequence[str], reduce: ScatterReduce = None) -> "ScatterSpec":
        return ScatterSpec.resolve(parse.Equation.parse(s), list(arr_names), reduce)
