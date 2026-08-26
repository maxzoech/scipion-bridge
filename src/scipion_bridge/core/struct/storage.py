from typing import Optional
from .schema import Schema


class SchemaArrayStorage:
    """Mix-in providing schema-based array storage inspection and operations."""

    schema: Schema

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    @classmethod
    def print_schema(cls) -> None:
        assert cls.schema is not None

        cls.schema.print_tree()