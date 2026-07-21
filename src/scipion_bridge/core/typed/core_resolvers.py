from .resolve import current_registry, resolver


@resolver
def resolve_any_to_str(value: object) -> str:
    return str(value)


@resolver
def resolve_tuple_to_str(value: tuple) -> str:
    return " ".join([current_registry().resolve(v, astype=str) for v in value])
