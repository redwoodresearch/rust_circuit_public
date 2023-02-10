from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Tuple, Literal, Callable

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Expr(ABC):
    @abstractmethod
    def render(self, out: list[str]):
        ...

    def __str__(self):
        out = []
        self.render(out)
        return "".join(out)

    def __repr__(self):
        return f"Expr.parse({str(self)!r})"

    @staticmethod
    def parse(s: str) -> "Expr":
        e, str_idx = parse_expr(StrIdx(s, 0))
        if e is None or not str_idx.done():
            raise ValueError(f"Invalid expression: {s!r}")
        return e


@dataclass(repr=False)
class IdentExpr(Expr):
    name: str
    idx: list[Expr]

    def render(self, out: list[str]):
        out.append(self.name)
        if self.idx:
            out.append("[")
            for i, e in enumerate(self.idx):
                if i > 0:
                    out.append(", ")
                e.render(out)
            out.append("]")


@dataclass(repr=False)
class IntExpr(Expr):
    value: int

    def render(self, out: list[str]):
        out.append(str(self.value))


if TYPE_CHECKING:
    EquationKind = Literal["=", "<-"]


@dataclass(repr=False)
class Equation:
    kind: EquationKind
    lhs: Expr
    rhs: Expr

    def render(self, out: list[str]):
        self.lhs.render(out)
        out.append(f" {self.kind} ")
        self.rhs.render(out)

    def __str__(self):
        out = []
        self.render(out)
        return "".join(out)

    def __repr__(self):
        return f"Equation.parse({str(self)!r})"

    @staticmethod
    def parse(s: str) -> "Equation":
        eq, str_idx = parse_equation(StrIdx(s, 0))
        if eq is None or not str_idx.done():
            raise ValueError(f"Invalid equation: {s!r}")
        return eq


@dataclass
class StrIdx:
    content: str
    idx: int

    def advance(self, n: int = 1) -> "StrIdx":
        return StrIdx(self.content, self.idx + n)

    def peek(self, n: int = 1) -> Optional[str]:
        if self.idx + n > len(self.content):
            return None
        return self.content[self.idx : self.idx + n]

    def next(self) -> Tuple[Optional[str], "StrIdx"]:
        c = self.peek()
        if c is None:
            return None, self
        return c, self.advance()

    def skip_whitespace(self) -> "StrIdx":
        while True:
            if self.peek() in [" ", "\t", "\n"]:
                self = self.advance()
            elif self.peek() == "#":
                while self.peek() not in ["\n", None]:
                    self = self.advance()
            else:
                break
        return self

    def done(self) -> bool:
        self = self.skip_whitespace()
        return self.peek() is None


def parse_ident(s: StrIdx) -> Tuple[Optional[str], StrIdx]:
    orig = s
    s = s.skip_whitespace()
    name: list[str] = []
    while (c := s.peek()) is not None and (c.isalnum() or c == "_"):
        if len(name) == 0 and not (c.isalpha() or c == "_"):
            return None, orig
        name.append(c)
        s = s.advance()
    if len(name) == 0:
        return None, orig
    return "".join(name), s


def is_ident(s: str) -> bool:
    ident = parse_ident(StrIdx(s, 0))[0]
    return ident is not None and ident == s


def parse_int(s: StrIdx) -> Tuple[Optional[int], StrIdx]:
    orig = s
    s = s.skip_whitespace()
    value: list[str] = []
    while (c := s.peek()) is not None and c.isdigit():
        value.append(c)
        s = s.advance()
    if len(value) == 0:
        return None, orig
    return int("".join(value)), s


def parse_expr(s: StrIdx) -> Tuple[Optional[Expr], StrIdx]:
    orig = s

    s = s.skip_whitespace()

    ident, s = parse_ident(s)
    if ident is not None:
        idx: list[Expr] = []
        s = s.skip_whitespace()
        if s.peek() != "[":
            return IdentExpr(ident, idx), s
        s = s.advance()
        while True:
            s = s.skip_whitespace()
            if s.peek() == "]":
                s = s.advance()
                break
            e, s = parse_expr(s)
            if e is None:
                return None, orig
            idx.append(e)
            s = s.skip_whitespace()
            if s.peek() == ",":
                s = s.advance()
            elif s.peek() != "]":
                return None, orig
        return IdentExpr(ident, idx), s

    value, s = parse_int(s)
    if value is not None:
        return IntExpr(value), s

    return None, orig


def parse_equation(s: StrIdx) -> Tuple[Optional[Equation], StrIdx]:
    orig = s
    lhs, s = parse_expr(s)
    if lhs is None:
        return None, orig
    s = s.skip_whitespace()
    kind: EquationKind
    if s.peek() == "=":
        kind = "="
    elif s.peek(2) == "<-":
        kind = "<-"
    else:
        return None, orig
    s = s.advance(len(kind))
    rhs, s = parse_expr(s)
    if rhs is None:
        return None, orig
    return Equation(kind, lhs, rhs), s
