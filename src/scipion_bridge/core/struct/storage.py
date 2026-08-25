
from .schema import Schema


class SchemaArrayStorage:

    def __init__(self, schema: Schema) -> None:

        self.schema = schema


    def print_schema(self):
        self.schema.print_tree()