import os
from pathlib import Path
import requests
import json
import gzip
import shutil

from collections import namedtuple

EMDB_EBI_REPOSITORY = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures"
EMDB_EBI_JSON_REPOSITORY = "https://www.ebi.ac.uk/emdb/api/entry"
PDB_EBI_REPOSITORY = "https://www.ebi.ac.uk/pdbe/entry-files/download"
EMDB_FTP_SERVER = "ftp.ebi.ac.uk"
EMDB_EBI_REPOSITORY_HALFMAPS = (
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/other/"
)

EMDBMetadata = namedtuple(
    "EMDBMetadata",
    ("pdb_id", "resolution", "sampling", "size", "org_x", "org_y", "org_z"),
)


def download_emdb_metadata(entry_id: int) -> EMDBMetadata:
    """
    Downloads metadata for an emdb entry

    Args:
        - entry: The EMDB ID to download. Use only the numbers after the 'EMD-' prefix
    """

    entry = f"EMD-{entry_id}"
    url = f"{EMDB_EBI_JSON_REPOSITORY}/{entry}"

    response = requests.get(url)
    response.raise_for_status()

    raw_data = json.loads(response.content)

    map_info = raw_data["map"]
    resolution = float(
        raw_data["structure_determination_list"]["structure_determination"][0][
            "image_processing"
        ][0]["final_reconstruction"]["resolution"]["valueOf_"]
    )

    try:
        pdb_id = raw_data["structure_determination_list"]["structure_determination"][0][
            "image_processing"
        ][0]["startup_model"][0]["pdb_model"]["pdb_id"]
        pdb_id = str(pdb_id).lower()
    except KeyError:
        pdb_id = None

    metadata = EMDBMetadata(
        pdb_id=pdb_id,
        resolution=resolution,
        sampling=float(map_info["pixel_spacing"]["y"]["valueOf_"]),
        size=int(map_info["dimensions"]["col"]),
        org_x=-(map_info["origin"]["col"]),
        org_y=-(map_info["origin"]["sec"]),
        org_z=-(map_info["origin"]["row"]),
    )

    return metadata


def _download(url, location: os.PathLike, force: bool):
    location = Path(location)
    if location.is_file() and force == False:
        return location

    response = requests.get(url)
    response.raise_for_status()

    with open(location, mode="wb") as file:
        file.write(response.content)

    return location


def _decompress(compressed: os.PathLike, dest_path: os.PathLike):
    with gzip.open(compressed, "rb") as f_in, open(dest_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return dest_path


def download_emdb_map(map_id, path: os.PathLike, force_download=False):
    path = Path(path)

    filename = f"emd_{map_id}.map"
    gz_filename = f"{filename}.gz"

    url = f"{EMDB_EBI_REPOSITORY}/EMD-{map_id}/map/{gz_filename}"
    dest_path_compressed = path / gz_filename
    dest_path = path / filename

    return _decompress(
        _download(url, dest_path_compressed, force=force_download), dest_path
    )


def _download_halfmap(
    emdb_entry, location: os.PathLike, half: int, force_download: bool
):
    halfmap_name = f"emd_{emdb_entry}_half_map_{half}.map"
    halfmap_name_compressed = halfmap_name + ".gz"

    halfmap_url = EMDB_EBI_REPOSITORY_HALFMAPS % emdb_entry + halfmap_name_compressed
    compressed_dest = location / halfmap_name_compressed
    halfmap_dest = location / halfmap_name

    return _decompress(
        _download(halfmap_url, compressed_dest, force=force_download), halfmap_dest
    )


def download_halfmaps(emdb_entry, location: os.PathLike, force_download=False):
    map_1 = _download_halfmap(
        emdb_entry, location, half=1, force_download=force_download
    )
    map_2 = _download_halfmap(
        emdb_entry, location, half=2, force_download=force_download
    )

    return map_1, map_2


def download_pdb_model(pdb_id, path: os.PathLike, force_download=False):
    path = Path(path)

    filename = f"pdb{pdb_id}.ent"

    url = f"{PDB_EBI_REPOSITORY}/pdb{pdb_id}.ent"
    dest_path = path / filename

    return _download(url, dest_path, force=force_download)
