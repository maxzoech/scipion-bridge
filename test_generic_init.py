from typing import Generic, TypeVar

T = TypeVar('T')

class SchemaConvertible:
    def __init__(self, *args, **kwargs):
        print("SchemaConvertible init")
        super().__init__(*args, **kwargs)
        self.configure_array_storage()

    def configure_array_storage(self):
        print("Storage configured")

class Set(Generic[T], SchemaConvertible):
    pass

s = Set[int]()
