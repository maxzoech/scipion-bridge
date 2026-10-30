"""Benchmark MRC stack chunk readers: mmap copy, scipion_bridge's sequential
reader (A) and vitrio (B, optional, needs the vitrio package). MRC_STACK names
the stack to read; it should be much larger than the memory of the machine.

Usage: MRC_STACK=stack.mrcs python bench_mrc_read.py OUT.jsonl [--readers mmap,A,B] [--conc 1,2,4]
                                [--chunks 1000,256] [--reps 3] [--seed 0]

/data is ZFS (raidz2 over sda-sdd): read() is served from the ARC, which
POSIX_FADV_DONTNEED does not drop. So the stack is probed for blocks that
are not in the ARC, every chunk of a cold read comes from such a block that
was never read before, and the bytes read from the pool's disks during a
measurement are recorded as disk_fraction (~1 = cold). The page cache is still evicted and checked with
mincore for the mmap reader. Warm reads repeat the same ranges.
"""

import argparse
import ctypes
import json
import mmap
import multiprocessing as mp
import os
import random
import sys
import time

import mrcfile
import numpy as np

PATH = os.environ["MRC_STACK"]
BUDGET = {1000: 2000, 256: 1024}  # images per measurement (mmap: half)
PAGE = mmap.PAGESIZE
BLOCK = 1024  # images; chunks are taken from fresh, ARC-cold blocks

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def header():
    with mrcfile.open(PATH, header_only=True) as m:
        h = m.header
        return (
            h.nbytes + int(h.nsymbt),
            int(h.nz),
            int(h.ny),
            int(h.nx),
        )


OFFSET, NZ, NY, NX = header()
ITEM = NY * NX * 4


def byte_range(start, count):
    begin = (OFFSET + start * ITEM) // PAGE * PAGE
    end = OFFSET + (start + count) * ITEM
    return begin, end - begin


def evict(start, count):
    begin, length = byte_range(start, count)
    fd = os.open(PATH, os.O_RDONLY)
    os.posix_fadvise(fd, begin, length, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def resident_pages(start, count):
    begin, length = byte_range(start, count)
    fd = os.open(PATH, os.O_RDONLY)
    addr = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, begin)
    assert addr not in (None, ctypes.c_void_p(-1).value), ctypes.get_errno()
    npages = (length + PAGE - 1) // PAGE
    vec = ctypes.create_string_buffer(npages)
    assert libc.mincore(addr, length, vec) == 0, ctypes.get_errno()
    libc.munmap(addr, length)
    os.close(fd)
    return sum(b & 1 for b in vec.raw)


def make_reader(name):
    match name:
        case "mmap":

            def read(start, count):
                with mrcfile.mmap(PATH, mode="r") as mrc:
                    return np.array(mrc.data[start : start + count], dtype=np.float32)

            return read

        case "A":
            from pathlib import Path
            from scipion_bridge.single_particle.resolvers import _read_stack

            def read(start, count):
                return _read_stack(Path(PATH), slice(start, start + count))

            return read

        case _ if name.startswith("B"):
            import vitrio as vio

            workers = int(name[1:]) if len(name) > 1 else None
            loader = vio.loader(workers=workers, cache=1)

            def read(start, count):
                out = np.empty((count, NY, NX), dtype=np.float32)
                locations = [
                    vio.ImageLocation(PATH, i) for i in range(start, start + count)
                ]
                vio.read_batch_async(loader, out, locations).get()
                return out

            return read

        case _:
            raise ValueError(name)


def worker(name, ranges, barrier, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    read = make_reader(name)
    for start, count in ranges:
        evict(start, count)
    resident = sum(resident_pages(s, c) for s, c in ranges)
    timings = {}
    outputs = []
    for phase in ("cold", "warm"):
        barrier.wait()
        t0 = time.perf_counter()
        outputs = [read(s, c) for s, c in ranges]
        timings[phase] = (t0, time.perf_counter())
    with mrcfile.mmap(PATH, mode="r") as mrc:
        identical = all(
            np.array_equal(out, np.asarray(mrc.data[s : s + c], dtype=np.float32))
            for out, (s, c) in zip(outputs, ranges)
        )
    for start, count in ranges:
        evict(start, count)
    queue.put((resident, timings, identical))


POOL_DISKS = ("sda", "sdb", "sdc", "sdd")  # raidz2 vdev of the 'data' pool


def disk_read_bytes():
    with open("/proc/diskstats") as f:
        rows = [line.split() for line in f]
    return sum(int(r[5]) * 512 for r in rows if r[2] in POOL_DISKS)


def is_cold(base, probes=8, threshold_s=1e-3):
    """True if every probed record of the block misses the ARC.

    A 4 KiB pread is served from the ARC in microseconds and from the disks
    in milliseconds. Each probe pulls one 128 KiB record into the ARC.
    """
    fd = os.open(PATH, os.O_RDONLY)
    times = []
    for k in range(probes):
        offset = OFFSET + (base + k * BLOCK // probes) * ITEM
        t0 = time.perf_counter()
        os.pread(fd, 4096, offset)
        times.append(time.perf_counter() - t0)
    os.close(fd)
    return min(times) > threshold_s


def measure(name, ranges_per_proc):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(ranges_per_proc))
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(name, r, barrier, queue))
        for r in ranges_per_proc
    ]
    disk0 = disk_read_bytes()
    for p in procs:
        p.start()
    results = [queue.get() for _ in procs]
    for p in procs:
        p.join()
    nimages = sum(c for r in ranges_per_proc for _, c in r)
    row = {
        "images": nimages,
        "disk_fraction": (disk_read_bytes() - disk0) / (nimages * ITEM),
        "resident_before_cold": sum(r[0] for r in results),
        "identical": all(r[2] for r in results),
    }
    for phase in ("cold", "warm"):
        t0 = min(r[1][phase][0] for r in results)
        t1 = max(r[1][phase][1] for r in results)
        row[f"{phase}_s"] = t1 - t0
        row[f"{phase}_MBps"] = nimages * ITEM / (t1 - t0) / 1e6
        row[f"{phase}_pps"] = nimages / (t1 - t0)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--readers", default="mmap,A,B")
    parser.add_argument("--conc", default="1,2,4")
    parser.add_argument("--chunks", default="1000,256")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    readers = args.readers.split(",")
    rng = random.Random(args.seed)
    # A fresh, disjoint 4096-image block of the 1M stack for every
    # measurement, readers in random order.
    blocks = [b for b in range(0, NZ - BLOCK, BLOCK) if is_cold(b, probes=4)]
    print(f"{len(blocks)} cold blocks of {BLOCK} images", file=sys.stderr, flush=True)
    rng.shuffle(blocks)

    def cold_starts(chunk, nchunks):
        # Every block holds BLOCK // chunk chunks; no chunk is read twice.
        per_block = BLOCK // chunk
        bases = [blocks.pop() for _ in range(-(-nchunks // per_block))]
        return [b + k * chunk for b in bases for k in range(per_block)][:nchunks]

    with open(args.out, "a") as out:
        for rep in range(args.reps):
            for chunk in map(int, args.chunks.split(",")):
                for conc in map(int, args.conc.split(",")):
                    for name in rng.sample(readers, len(readers)):
                        budget = BUDGET[chunk] // 2 if name == "mmap" else BUDGET[chunk]
                        nchunks = max(budget // chunk, conc)
                        starts = cold_starts(chunk, nchunks)
                        ranges = [
                            [(s, chunk) for s in starts[p::conc]] for p in range(conc)
                        ]
                        row = measure(name, ranges)
                        row.update(
                            reader=name,
                            chunk=chunk,
                            conc=conc,
                            rep=rep,
                            first=starts[0],
                        )
                        print(json.dumps(row), flush=True)
                        out.write(json.dumps(row) + "\n")
                        out.flush()


if __name__ == "__main__":
    main()
