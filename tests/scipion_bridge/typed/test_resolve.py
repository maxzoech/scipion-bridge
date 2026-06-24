import sys
import logging

import scipion_bridge as sb
import scipion_bridge.core.typed.resolve as resolve
from scipion_bridge.core.typed.resolve import ScopedPathfindingContainer as Container

import pytest


def test_basic_resolve():

    logging.getLogger().setLevel(logging.INFO)

    registry = resolve.Registry()
    registry.add_resolver(object, str, lambda x: str(x))
    registry.add_resolver(float, int, lambda x: int(x))

    # Needs the "scipion_bridge.typed" namespace to find the "_downcast" transformation
    func = registry.find_resolve_func(
        {"scipion_bridge.core.typed", __name__}, float, str, local_scope_name=__name__
    )
    resolved = func(2.5)

    assert resolved == "2.5"


def test_resolve_faulty_resolver():
    registry = resolve.Registry()
    registry.add_resolver(object, str, lambda x: int(x))  # Returns wrong type here
    registry.add_resolver(float, int, lambda x: int(x))

    with pytest.raises(TypeError):
        func = registry.find_resolve_func(
            {__name__}, float, str, local_scope_name="__main__"
        )
        _ = func(2.5)


def test_unresolvable_types_error():
    registry = resolve.Registry()
    registry.add_resolver(float, int, lambda x: int(x))  # Returns wrong type here
    registry.add_resolver(bool, int, lambda x: int(x))

    with pytest.raises(TypeError):
        func = registry.find_resolve_func(
            {__name__}, float, bool, local_scope_name="__main__"
        )
        _ = func(2.5)

    with pytest.raises(TypeError):
        func = registry.find_resolve_func(
            {__name__}, float, str, local_scope_name="__main__"
        )
        _ = func(2.5)


def test_resolved_func():

    @sb.resolver
    def resolve_float_to_int(value: float) -> int:
        return int(value)

    @sb.resolve_params
    def foo(bar: sb.Resolve[str, int], number: float, value):
        assert bar == "10"
        assert number == 42.0
        assert value == "Test"

    foo(10, number=42.0, value="Test")
    foo(10.0, number=42.0, value="Test")


def test_resolve_func_default_params():

    @sb.resolve_params
    def foo(bar: sb.Resolve[str] = 10):
        assert bar == "10"

    foo()


def test_resolve_passthrough():
    @sb.resolve_params
    def foo(bar: sb.Resolve[int, float]):
        assert bar == 42

    foo(42)


def test_pathfinding_container_ordering():
    """
    # Module executed from command line, so __name__ == "__main__"

    ### other_module ###
    def resolver_generic(value: A) -> B
        ...

    ### other_module.foo ###
    def resolver(value: A) -> B
        ...

    ### __main___ ###
    _downcast(A) -> C

    In this case we want to use the resolver in other_module.foo because it is
    the most specific one with the lowest weight. We don't select the other ones
    because:
    - _downcast exists in the local scope but has a higher weight
    - resolver_generic(value: A) -> B in other_module is shadowed by other_module.foo
    """

    candidates = [
        Container(
            "B",
            None,
            weight=0,
            incoming_edge_attributes=Container.ResolverNode(
                FnStub("resolver_generic"), "other_module"
            ),
            local_scope_name="__main__",
        ),
        Container(
            "B",
            None,
            weight=0,
            incoming_edge_attributes=Container.ResolverNode(
                FnStub("resolver"), "other_module.foo"
            ),
            local_scope_name="__main__",
        ),
        Container(
            "C",
            None,
            weight=1,
            incoming_edge_attributes=Container.ResolverNode(
                FnStub("_downcast"), "__main__"
            ),
            local_scope_name="__main__",
        ),
    ]

    sorted_candidates = sorted(candidates, reverse=False)
    assert sorted_candidates[0].edge_attributes.resolver_fn.name == "resolver"  # type: ignore


class FnStub:

    def __init__(self, name) -> None:
        self.name = name
        self.__qualname__ = name


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11 or higher")
def test_pathfinding_container_ordering_local_scope_shadowing():
    """
    # Module executed from command line, so __name__ == "__main__"

    ### other_module ###
    def resolver_generic(value: A) -> B
        ...

    ### other_module.foo ###
    def resolver(value: A) -> B
        ...

    ### __main___ ###
    _downcast(A) -> C

    def resolve_local(value: A) -> B
        ...

    ### __main___.bar ###
    def resolve_local_specific(value: A) -> B
        ...

    In this case we want to use the resolver in __main__.bar because it is
    the most specific in the local scope. We don't select the other ones
    because:
    - _downcast exists in the local scope but has a higher weight
    - The other ones are shadowed by __main__.bar
    """

    candidates = [
        Container(
            "B",
            None,
            weight=0,
            incoming_edge_attributes=Container.ResolverNode(
                FnStub("resolver_generic"), "other_module"
            ),
            local_scope_name="__main__",
        ),
        Container(
            "B",
            None,
            weight=0,
            incoming_edge_attributes=Container.ResolverNode(
                FnStub("resolver"), "other_module.foo"
            ),
            local_scope_name="__main__",
        ),
        Container(
            "C",
            None,
            weight=1,
            incoming_edge_attributes=Container.ResolverNode(
                FnStub("_downcast"), "__main__"
            ),
            local_scope_name="__main__",
        ),
        Container(
            "B",
            None,
            weight=0,
            incoming_edge_attributes=Container.ResolverNode(
                FnStub("resolve_local"), "__main__"
            ),
            local_scope_name="__main__",
        ),
        Container(
            "B",
            None,
            weight=0,
            incoming_edge_attributes=Container.ResolverNode(
                FnStub("resolve_local_specific"), "__main__.bar"
            ),
            local_scope_name="__main__",
        ),
    ]

    sorted_candidates = sorted(candidates, reverse=False)
    assert sorted_candidates[0].edge_attributes.resolver_fn.name == "resolve_local_specific"  # type: ignore


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11 or higher")
def test_resolve_namespaces():

    def bar():
        @sb.resolver
        def resolve_float(value: float) -> str:
            return str(value * 2)

        @sb.resolve_params
        def foo(bar: sb.Resolve[str]):
            return bar

        r = foo(42.0)
        return r

    @sb.resolve_params
    def foo(bar: sb.Resolve[str]):
        return bar

    r = bar()
    assert r == "84.0"

    r = foo(42.0)
    assert r == "42.0"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11 or higher")
def test_resolve_namespaces_recursive():

    # logging.basicConfig(level=logging.INFO)

    @sb.resolver
    def resolve_tuple_to_str_underline(value: tuple) -> str:
        return "_".join(
            [sb.resolve(v, astype=str) for v in value]
        )

    def bar():
        @sb.resolver
        def resolve_float(value: float) -> str:
            return str(value * 2)

        @sb.resolve_params
        def foo(bar: sb.Resolve[str]):
            return bar

        r = foo((42.0, 40.0, 5.0))
        return r

    @sb.resolve_params
    def foo(bar: sb.Resolve[str]):
        return bar

    r = bar()
    assert r == "84.0_80.0_10.0"

    r = foo((42.0, 40.0, 5.0))
    assert r == "42.0_40.0_5.0"


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    test_resolve_namespaces_recursive()
