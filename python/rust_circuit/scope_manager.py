from __future__ import annotations

from copy import copy
from typing import Iterable, Optional, Set, Tuple, Union

import attrs
from attrs import define, frozen

import rust_circuit as rc
from rust_circuit.py_utils import assert_never


@frozen
class ScopeGlobalTraversal:
    traversal: rc.IterativeMatcher


@frozen
class ScopeMatcher:
    matcher: rc.IterativeMatcher


ScopeUpdate = Union[ScopeMatcher, ScopeGlobalTraversal]
NamedUpdates = Tuple[Tuple[ScopeUpdate, Optional[str]], ...]


@define
class ScopeManager:
    """
    Wrapper around a `circuit` for the purposes of doing rewrites

    Note that self.unique() is the circuit focused on, but self.circuit may be
    a parent in a larger computational graph
    """

    circuit: rc.Circuit
    # TODO: add nice stuff for working with names if that seems nice
    named_updates: NamedUpdates = ()
    assert_matched_exists: bool = True
    assert_matched_unique: bool = False

    def __attrs_post_init__(self):
        self.check()

    def check(self):
        matched = self.circuit.get(self.matcher())
        if self.assert_matched_exists:
            assert len(matched) > 0
        if self.assert_matched_unique:
            assert len(matched) == 1

    def raw_matcher(self, *extra: rc.IterativeMatcherIn):
        return rc.IterativeMatcher.new_chain(*self.matchers, *extra)

    def traversal(self, *extra_traversals: rc.IterativeMatcherIn):
        return rc.IterativeMatcher.all(*self.traversals, *extra_traversals)

    def matcher(self, *extra: rc.IterativeMatcherIn, extra_traversals: Iterable[rc.IterativeMatcherIn] = []):
        return self.raw_matcher(*extra) & self.traversal(*extra_traversals)

    @property
    def c(self):
        return self.circuit

    @property
    def matchers(self):
        yield rc.restrict(rc.IterativeMatcher(True), term_if_matches=True)
        for g_m, _ in self.named_updates:
            if isinstance(g_m, ScopeMatcher):
                yield g_m.matcher

    @property
    def traversals(self):
        for g_m, _ in self.named_updates:
            if isinstance(g_m, ScopeGlobalTraversal):
                yield g_m.traversal

    def sub_get_new_updates(self, sub_update: ScopeUpdate, name: Optional[str] = None) -> NamedUpdates:
        """get new named_updates
        sub_item is either node getter for chain or term_early_at item"""
        return self.named_updates + ((sub_update, name),)

    def sub_update(self, sub_update: ScopeUpdate, name: Optional[str] = None) -> ScopeManager:
        """evolve new instance with new named_updates (see sub_u)"""
        return attrs.evolve(self, named_updates=self.sub_get_new_updates(sub_update, name=name))

    def sub_update_(self, sub_update: ScopeUpdate, name: Optional[str] = None):
        """mutably set to new named_updates (see sub_u)"""
        new_named_updates = self.sub_get_new_updates(sub_update, name=name)
        attrs.evolve(self, named_updates=new_named_updates)  # run check
        self.named_updates = new_named_updates

    def sub_matcher(self, matcher: rc.IterativeMatcherIn, name: Optional[str] = None):
        """wrap sub"""
        return self.sub_update(ScopeMatcher(rc.IterativeMatcher(matcher)), name=name)

    def sub_matcher_(self, matcher: rc.IterativeMatcherIn, name: Optional[str] = None):
        """wrap sub_"""
        return self.sub_update_(ScopeMatcher(rc.IterativeMatcher(matcher)), name=name)

    def sub_traversal(self, traversal: rc.IterativeMatcherIn, name: Optional[str] = None):
        """wrap sub"""
        return self.sub_update(ScopeGlobalTraversal(rc.IterativeMatcher(traversal)), name=name)

    def sub_traversal_(self, traversal: rc.IterativeMatcherIn, name: Optional[str] = None):
        """wrap sub_"""
        return self.sub_update_(ScopeGlobalTraversal(rc.IterativeMatcher(traversal)), name=name)

    def remove_by_name(self, name: str):
        return tuple((i, s) for i, s in self.named_updates if s != name)

    def pop_sub_by_name_(self, name: str):
        self.named_updates = self.remove_by_name(name)

    def pop_sub_by_name(self, name: str) -> ScopeManager:
        return attrs.evolve(self, named_updates=self.remove_by_name(name))

    def pop_sub_(self):
        self.named_updates = self.named_updates[:-1]

    def pop_sub(self) -> ScopeManager:
        return attrs.evolve(self, named_updates=self.named_updates[:-1])

    def clear_sub_(self):
        self.named_updates = ()

    def clear_sub(self) -> ScopeManager:
        return attrs.evolve(self, named_updates=())

    def clone(self):
        return copy(self)

    def get_bound_updater(
        self,
        updater: Union[rc.Updater, rc.BoundUpdater, rc.TransformIn],
        apply_global: bool = False,
    ):
        if not isinstance(updater, (rc.Updater, rc.BoundUpdater)):
            updater = rc.Updater(updater)
        if not isinstance(updater, rc.BoundUpdater):
            if apply_global:
                return updater.bind(rc.IterativeMatcher.term(match_next=True))
            else:
                return updater.bind(self.matcher())
        if isinstance(updater, rc.BoundUpdater):
            return self.matcher(updater.matcher).updater(updater.updater.transform)
        else:
            assert_never(updater)

    def update(
        self,
        updater: Union[rc.Updater, rc.BoundUpdater, rc.TransformIn],
        apply_global: bool = False,
    ) -> ScopeManager:
        """evolve new instance with updated circuit (see get_bound_updater)"""
        return attrs.evolve(self, circuit=self.get_bound_updater(updater, apply_global=apply_global)(self.circuit))

    def update_(
        self,
        updater: Union[rc.Updater, rc.BoundUpdater, rc.TransformIn],
        apply_global: bool = False,
    ) -> None:
        """mutably set to new updated circuit (see get_bound_updater)"""
        self.circuit = self.get_bound_updater(updater, apply_global=apply_global)(self.circuit)

    def print_path(
        self,
        printer: rc.PrintOptionsBase = rc.PrintOptions(bijection=False, colorer=lambda _: 3),
        path_from: rc.IterativeMatcherIn = rc.IterativeMatcher.term(match_next=True),
    ):
        """Prints the path to the currently examined nodes.

        Use the colorer of the print passed as argument to color matched nodes"""
        matched = self.matched_circuits()
        path_printer = (
            printer.evolve(
                traversal=self.traversal(rc.new_traversal(term_early_at=~rc.Matcher.match_any_found(matched))),
                colorer=lambda x: (printer.colorer(x) if x in matched else None),  # type: ignore
                bijection=False,
            )
            if isinstance(printer, rc.PrintOptions)
            else printer.evolve(
                traversal=self.traversal(rc.new_traversal(term_early_at=~rc.Matcher.match_any_found(matched))),
                colorer=lambda x: (printer.colorer(x) if x in matched else None),  # type: ignore
            )
        )

        print()
        path_printer.print(rc.Getter().get_unique(self.circuit, path_from))  # type: ignore

    def print(self, printer: rc.PrintOptionsBase = rc.PrintOptions(bijection=False)):
        # TODO: support HTML printer
        printer = (
            printer.evolve(traversal=self.traversal(printer.traversal), bijection=False)
            if isinstance(printer, rc.PrintOptions)
            else printer.evolve(traversal=self.traversal(printer.traversal))
        )
        for c in self.matched_circuits():
            print()
            printer.print(c)

    def unique(self) -> rc.Circuit:
        return self.matcher().get_unique(self.circuit)

    def matched_circuits(self) -> Set[rc.Circuit]:
        return self.matcher().get(self.circuit)
