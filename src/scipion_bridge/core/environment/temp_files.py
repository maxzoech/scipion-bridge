import os
from pathlib import Path
import tempfile
from typing import Optional

import logging


class TemporaryFilesProvider:

    def new_temporary_file(
        self, suffix: Optional[str] = None, prefix: Optional[str] = None
    ) -> os.PathLike:
        new_file = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, delete=False
        ).name
        logging.debug(f"Creating new temporary file at {new_file}")

        return Path(new_file)

    def delete(self, path: os.PathLike):
        logging.debug(f"Remove file at {path}")

        os.remove(path)
