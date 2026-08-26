from typing import Optional
from .schema import Schema


class SchemaArrayStorage:
    """Mix-in providing schema-based array storage inspection and operations."""

    schema: Schema

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def print_schema(self) -> None:
        assert self.schema is not None

        self.schema.print_tree()