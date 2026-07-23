import zarr
import numpy as np

store1 = zarr.storage.MemoryStore()
group1 = zarr.group(store=store1)
source = group1.create_array('source', shape=(100, 100), dtype='f8', chunks=(10, 100))
source[:] = np.ones((100, 100))

store2 = zarr.storage.MemoryStore()
group2 = zarr.group(store=store2)
target = group2.create_array('target', shape=(10, 100, 100), dtype='f8', chunks=(1, 10, 100))

# Try block-by-block manually
try:
    chunk_size = source.chunks[0]
    for i in range(0, source.shape[0], chunk_size):
        chunk_data = source[i:i+chunk_size]
        target[0, i:i+chunk_size] = chunk_data
    print("Block-by-block assignment worked")
except Exception as e:
    print(f"Block-by-block assignment failed: {type(e).__name__}: {e}")

