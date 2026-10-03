import pytest
import scipion_bridge as sb
import scipion_bridge.core.typed.resolve as resolve
from scipion_bridge.core.typed.resolve import ScopedPathfindingContainer as Container


class SourceType:
    def __init__(self, data: list[int]) -> None:
        self.data = data


class IntermediateType:
    def __init__(self, data: list[int]) -> None:
        self.data = data


class TargetType:
    def __init__(self, data: list[int]) -> None:
        self.data = data


def test_function_projector_creation():
    """Verify that a function decorated with @resolver gains .forward and remains callable."""

    class TypeA:
        def __init__(self, x: int) -> None:
            self.x = x

    class TypeB:
        def __init__(self, x: int) -> None:
            self.x = x

    @sb.resolver
    def convert_a_to_b(value: TypeA) -> TypeB:
        return TypeB(value.x * 2)

    # 1. Stays callable directly
    res = convert_a_to_b(TypeA(5))
    assert isinstance(res, TypeB)
    assert res.x == 10

    # 2. Has .forward
    res_fwd = convert_a_to_b.forward(TypeA(7))
    assert isinstance(res_fwd, TypeB)
    assert res_fwd.x == 14


def test_class_resolver_forward_method():
    """Verify that a class defining forward decorated with @resolver registers and executes."""

    class TypeC:
        def __init__(self, text: str) -> None:
            self.text = text

    class TypeD:
        def __init__(self, text: str) -> None:
            self.text = text

    @sb.resolver
    class CToDResolver:
        def forward(self, value: TypeC) -> TypeD:
            return TypeD(value.text.upper())

    # Direct instantiation & call
    resolver_inst = CToDResolver()
    assert resolver_inst(TypeC("hello")).text == "HELLO"
    assert resolver_inst.forward(TypeC("world")).text == "WORLD"

    # End-to-end resolution via B.resolve
    resolved = sb.resolve(TypeC("scipion"), astype=TypeD)
    assert isinstance(resolved, TypeD)
    assert resolved.text == "SCIPION"


def test_guardrail_rejects_class_iter():
    """Verify that registering a class with __iter__ raises TypeError immediately."""

    class TypeE:
        pass

    class TypeF:
        pass

    with pytest.raises(TypeError, match="must not implement '__iter__'"):

        @sb.resolver
        class InvalidIterResolver:
            def forward(self, value: TypeE) -> TypeF:
                return TypeF()

            def __iter__(self):
                yield 1


def test_guardrail_rejects_class_without_forward():
    """Verify that registering a class without forward (e.g. only __call__) raises TypeError."""

    class TypeG:
        pass

    class TypeH:
        pass

    with pytest.raises(TypeError, match="must define a 'forward' method"):

        @sb.resolver
        class InvalidCallOnlyResolver:
            def __call__(self, value: TypeG) -> TypeH:
                return TypeH()


def test_composed_resolver_monolithic():
    """Verify that ComposedResolver executes multi-step paths monotonically."""
    registry = resolve.Registry()

    # Path: SourceType -> IntermediateType -> TargetType
    def step1(value: SourceType) -> IntermediateType:
        return IntermediateType([x + 1 for x in value.data])

    def step2(value: IntermediateType) -> TargetType:
        return TargetType([x * 2 for x in value.data])

    registry.add_resolver(SourceType, IntermediateType, step1, namespace="test_mod")
    registry.add_resolver(IntermediateType, TargetType, step2, namespace="test_mod")

    composed = registry.find_resolve_func(
        {"test_mod"}, SourceType, TargetType, local_scope_name="test_mod"
    )

    assert isinstance(composed, resolve.ComposedResolver)
    result = composed(SourceType([1, 2, 3]))
    assert isinstance(result, TargetType)
    # (x + 1) * 2: [ (1+1)*2, (2+1)*2, (3+1)*2 ] = [4, 6, 8]
    assert result.data == [4, 6, 8]


def test_composed_resolver_slice_narrowing():
    """
    Verify the narrowing rule:
    Step 1 (Source -> Intermediate) does NOT take slice -> passes data downstream.
    Step 2 (Intermediate -> Target) takes slice -> consumes slice, resets current_slice to None.
    """
    registry = resolve.Registry()

    def step1_noslice(value: SourceType) -> IntermediateType:
        return IntermediateType([x * 10 for x in value.data])

    class SliceableStep2:
        def forward(
            self, value: IntermediateType, *, slice: resolve.Optional[slice] = None
        ) -> TargetType:
            data = value.data[slice] if slice is not None else value.data
            return TargetType(data)

    registry.add_resolver(
        SourceType, IntermediateType, step1_noslice, namespace="test_mod"
    )
    registry.add_resolver(
        IntermediateType, TargetType, SliceableStep2, namespace="test_mod"
    )

    composed = registry.find_resolve_func(
        {"test_mod"}, SourceType, TargetType, local_scope_name="test_mod"
    )

    # 1. Monolithic call (slice=None)
    full_res = composed(SourceType([0, 1, 2, 3, 4]))
    assert full_res.data == [0, 10, 20, 30, 40]

    # 2. Sliced call (slice=slice(1, 4))
    sliced_res = composed(SourceType([0, 1, 2, 3, 4]), slice=slice(1, 4))
    assert sliced_res.data == [10, 20, 30]


def test_composed_resolver_unconsumed_slice_raises():
    """Verify that requesting a slice when no resolver supports it fails fast with TypeError."""
    registry = resolve.Registry()

    def step1(value: SourceType) -> TargetType:
        return TargetType(value.data)

    registry.add_resolver(SourceType, TargetType, step1, namespace="test_mod")

    composed = registry.find_resolve_func(
        {"test_mod"}, SourceType, TargetType, local_scope_name="test_mod"
    )

    with pytest.raises(TypeError, match="no resolver along the path supports slicing"):
        composed(SourceType([1, 2, 3]), slice=slice(0, 2))


def test_dijkstra_slice_priority_tie_breaker():
    """Verify that when path weight and lexical scope are equal, Dijkstra prefers the sliceable edge."""
    registry = resolve.Registry()

    class SlicedTarget(TargetType):
        pass

    class NonSlicedTarget(TargetType):
        pass

    def resolver_nonslice(value: SourceType) -> TargetType:
        return TargetType(value.data)

    def resolver_slice(
        value: SourceType, *, slice: resolve.Optional[slice] = None
    ) -> TargetType:
        return TargetType(value.data if slice is None else value.data[slice])

    # Both in same namespace 'test_mod' with weight 0
    registry.add_resolver(
        SourceType, TargetType, resolver_nonslice, namespace="test_mod"
    )
    # Re-register with slice support in distinct subclass or edge
    # Let's test with Container sorting directly:
    container_nonslice = Container(
        TargetType,
        SourceType,
        weight=0,
        incoming_edge_attributes=Container.ResolverNode(
            resolver_nonslice, "test_mod", requires_slice=False
        ),
        local_scope_name="test_mod",
    )

    container_slice = Container(
        TargetType,
        SourceType,
        weight=0,
        incoming_edge_attributes=Container.ResolverNode(
            resolver_slice, "test_mod", requires_slice=True
        ),
        local_scope_name="test_mod",
    )

    # container_slice should be strictly less than container_nonslice (< min-heap priority)
    assert container_slice < container_nonslice
    assert not (container_nonslice < container_slice)


def test_dijkstra_preserves_lexical_scoping():
    """Verify that local scope ALWAYS takes precedence over external scope, even if external is sliceable."""

    def local_nonslice(value: SourceType) -> TargetType:
        return TargetType(value.data)

    def external_slice(
        value: SourceType, *, slice: resolve.Optional[slice] = None
    ) -> TargetType:
        return TargetType(value.data if slice is None else value.data[slice])

    local_container = Container(
        TargetType,
        SourceType,
        weight=0,
        incoming_edge_attributes=Container.ResolverNode(
            local_nonslice, "my_local_module", requires_slice=False
        ),
        local_scope_name="my_local_module",
    )

    external_container = Container(
        TargetType,
        SourceType,
        weight=0,
        incoming_edge_attributes=Container.ResolverNode(
            external_slice, "external_pkg.plugin", requires_slice=True
        ),
        local_scope_name="my_local_module",
    )

    # Local scope container MUST win (<) over external scope container!
    assert local_container < external_container
    assert not (external_container < local_container)


class SizedIntermediate:
    def __init__(self, data: list[int]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    @property
    def estimated_item_nbytes(self) -> int:
        return 100


class UnsizedIntermediate:
    def __init__(self, data: list[int]) -> None:
        self.data = data


def test_composed_resolver_iter():
    """Verify that ComposedResolver.iter slices through the first sliceable step in chunks."""
    registry = resolve.Registry()

    def step1(value: SourceType) -> SizedIntermediate:
        return SizedIntermediate([x * 2 for x in value.data])

    class SliceStep:
        def forward(
            self,
            value: SizedIntermediate,
            *,
            slice: resolve.Optional[slice] = None,
        ) -> TargetType:
            sub = value.data[slice] if slice is not None else value.data
            return TargetType(sub)

    registry.add_resolver(SourceType, SizedIntermediate, step1, namespace="test_mod")
    registry.add_resolver(
        SizedIntermediate, TargetType, SliceStep, namespace="test_mod"
    )

    composed = registry.find_resolve_func(
        {"test_mod"}, SourceType, TargetType, local_scope_name="test_mod"
    )

    items = list(range(10))
    chunks = list(composed.iter(SourceType(items), chunk_size=4))

    assert len(chunks) == 3
    assert [c.data for c in chunks] == [
        [0, 2, 4, 6],
        [8, 10, 12, 14],
        [16, 18],
    ]


def test_composed_resolver_iter_auto_chunk_size():
    """Verify that ComposedResolver.iter auto-tunes chunk_size targeting target_bytes."""
    registry = resolve.Registry()

    def step1(value: SourceType) -> SizedIntermediate:
        return SizedIntermediate(list(value.data))

    class SliceStep:
        def forward(
            self,
            value: SizedIntermediate,
            *,
            slice: resolve.Optional[slice] = None,
        ) -> TargetType:
            return TargetType(value.data[slice] if slice is not None else value.data)

    registry.add_resolver(SourceType, SizedIntermediate, step1, namespace="test_mod")
    registry.add_resolver(
        SizedIntermediate, TargetType, SliceStep, namespace="test_mod"
    )

    composed = registry.find_resolve_func(
        {"test_mod"}, SourceType, TargetType, local_scope_name="test_mod"
    )

    # 10 items, each 100 bytes. With target_bytes=350, chunk_size = 350 // 100 = 3
    chunks = list(
        composed.iter(
            SourceType(list(range(10))),
            target_bytes=350,
        )
    )
    assert len(chunks) == 4
    assert [len(c.data) for c in chunks] == [3, 3, 3, 1]


def test_composed_resolver_iter_no_sliceable_step_raises():
    """Verify that iter fails fast with TypeError if no step supports slicing."""
    registry = resolve.Registry()

    def step(value: SourceType) -> TargetType:
        return TargetType(value.data)

    registry.add_resolver(SourceType, TargetType, step, namespace="test_mod")

    composed = registry.find_resolve_func(
        {"test_mod"}, SourceType, TargetType, local_scope_name="test_mod"
    )

    with pytest.raises(
        TypeError,
        match="no resolver along the path supports slicing",
    ):
        composed.iter(SourceType([1, 2, 3]))


def test_composed_resolver_iter_invalid_chunk_size_raises():
    """Verify that invalid chunk_size <= 0 raises ValueError."""
    registry = resolve.Registry()

    class SliceStep:
        def forward(
            self,
            value: SizedIntermediate,
            *,
            slice: resolve.Optional[slice] = None,
        ) -> TargetType:
            return TargetType(value.data[slice] if slice is not None else value.data)

    registry.add_resolver(
        SizedIntermediate, TargetType, SliceStep, namespace="test_mod"
    )
    composed = registry.find_resolve_func(
        {"test_mod"}, SizedIntermediate, TargetType, local_scope_name="test_mod"
    )

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        composed.iter(SizedIntermediate([1, 2]), chunk_size=0)

    with pytest.raises(ValueError, match="target_bytes must be positive"):
        composed.iter(SizedIntermediate([1, 2]), target_bytes=-10)


def test_composed_resolver_iter_unsized_intermediate_raises():
    """Verify that an intermediate without __len__ raises TypeError upon iteration."""
    registry = resolve.Registry()

    def step1(value: SourceType) -> UnsizedIntermediate:
        return UnsizedIntermediate(value.data)

    class SliceStep:
        def forward(
            self,
            value: UnsizedIntermediate,
            *,
            slice: resolve.Optional[slice] = None,
        ) -> TargetType:
            return TargetType(value.data[slice] if slice is not None else value.data)

    registry.add_resolver(SourceType, UnsizedIntermediate, step1, namespace="test_mod")
    registry.add_resolver(
        UnsizedIntermediate, TargetType, SliceStep, namespace="test_mod"
    )

    composed = registry.find_resolve_func(
        {"test_mod"}, SourceType, TargetType, local_scope_name="test_mod"
    )

    it = composed.iter(SourceType([1, 2, 3]))
    with pytest.raises(TypeError, match="object does not define __len__"):
        next(it)


def test_find_resolver_precomputation(tmp_path):
    """Verify that B.find_resolver precomputes a ComposedResolver for types and instances."""
    import scipion_bridge as B
    import mrcfile
    import pandas as pd
    import starfile
    import numpy as np
    from pathlib import Path
    from scipion_bridge.single_particle.particle import Particle

    mrc_path = tmp_path / "particles.mrcs"
    star_path = tmp_path / "particles.star"
    n_particles = 6
    data = np.arange(n_particles * 8 * 8, dtype=np.float32).reshape(n_particles, 8, 8)
    with mrcfile.new(mrc_path) as mrc:
        mrc.set_data(data)

    df = pd.DataFrame(
        {"rlnImageName": [f"{i + 1:06d}@particles.mrcs" for i in range(n_particles)]},
    )
    starfile.write(df, star_path)

    # 1. Precompute resolver using types (cold path)
    resolver = B.find_resolver(Path, B.Set[Particle])
    assert isinstance(resolver, B.ComposedResolver)
    assert any(step.requires_slice for step in resolver.steps)

    # 2. Execute on instance (hot path - zero pathfinding)
    monolithic = resolver(star_path)
    assert len(monolithic) == 6

    # 3. Stream chunks via precomputed resolver
    chunks = list(resolver.iter(star_path, chunk_size=2))
    assert len(chunks) == 3
    assert all(len(c) == 2 for c in chunks)
    assert np.allclose(np.asarray(chunks[0]["pixels"]), data[0:2])
