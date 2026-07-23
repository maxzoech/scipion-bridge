import zarr
import numpy as np

store1 = zarr.storage.MemoryStore()
group1 = zarr.group(store=store1)
source = group1.create_array('source', shape=(100, 100), dtype='f8')
source[:] = np.ones((100, 100))

store2 = zarr.storage.MemoryStore()
group2 = zarr.group(store=store2)
target = group2.create_array('target', shape=(10, 100, 100), dtype='f8')

try:
    target[0] = source
    print("Direct assignment worked")
except Exception as e:
    print(f"Direct assignment failed: {type(e).__name__}: {e}")

