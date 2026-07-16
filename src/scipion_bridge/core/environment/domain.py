from dataclasses import dataclass
from typing import List


@dataclass
class Domain:
    name: str
    command: List[str]
    isolated: bool = False

    @classmethod
    def default(cls) -> "Domain":
        return Domain(
            name="Default",
            command=[],
            isolated=False
        )