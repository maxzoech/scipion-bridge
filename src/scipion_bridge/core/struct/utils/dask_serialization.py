from __future__ import annotations

import pyarrow as pa
from ..set import Set


def serialize_set(s: Set):
    header = {
        "dtype": s.dtype,
        "capacity": s.capacity,
        "length": len(s),
    }
    batch = s.to_arrow()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    frames = [sink.getvalue()]
    return header, frames


def deserialize_set(header, frames):
    (arrow_buffer,) = frames
    reader = pa.ipc.open_stream(arrow_buffer)
    batch = reader.read_next_batch()
    return Set.from_arrow(header["dtype"], batch)


def register_dask_serialization() -> bool:
    """Register custom zero-copy Arrow IPC serialization for Set with Dask Distributed."""
    try:
        import dask.distributed.protocol as dask_protocol
    except ImportError:
        return False

    dask_protocol.register_serialization(Set, serialize_set, deserialize_set)
    return True


# Auto-register if dask is installed
register_dask_serialization()

