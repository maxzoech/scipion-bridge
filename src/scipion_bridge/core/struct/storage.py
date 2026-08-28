import abc

from typing import Optional
from .schema import Schema


class SchemaArrayStorage(metaclass=abc.ABCMeta):

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def schema(cls) -> Schema:
        ...

    @classmethod
    def print_schema(cls) -> None:
        assert cls.schema is not None

        cls.schema().print_tree()