"""The hashing of the library: fingerprints, item ids and element ids.

Parts are joined unambiguously, each prefixed with its length, and hashed
with ``xxh3_128``, the hash the row digests of references use too. It is
fast and its 128 bits make accidental collisions negligible; none of these
hashes needs to resist an attacker.
"""

import xxhash

# Bytes of an id (xxh3, 128 bits).
ID_BYTES = 16


def join(*parts: bytes) -> bytes:
    """Concatenate ``parts`` unambiguously, each prefixed with its length."""
    return b"".join(len(part).to_bytes(8, "big") + part for part in parts)


def digest(*parts: bytes) -> str:
    """The hex digest of ``parts``, e.g. a fingerprint."""
    return xxhash.xxh3_128_hexdigest(join(*parts))


def item_id(*parts: bytes) -> bytes:
    """The 16-byte digest of ``parts``, e.g. the id of an item or element."""
    return xxhash.xxh3_128_digest(join(*parts))
