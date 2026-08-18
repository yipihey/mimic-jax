"""Reader for the legacy L-Halo binary merger-tree format used by Mini-Millennium."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# This dtype deliberately spells out the aligned C layout of ``struct RawHalo``.
# Its 104-byte item size is part of the on-disk Mini-Millennium contract.
LHALO_DTYPE = np.dtype(
    {
        "names": [
            "Descendant",
            "FirstProgenitor",
            "NextProgenitor",
            "FirstHaloInFOFgroup",
            "NextHaloInFOFgroup",
            "Len",
            "M_Mean200",
            "M_Crit200",
            "M_TopHat",
            "Pos",
            "Vel",
            "VelDisp",
            "Vmax",
            "Spin",
            "MostBoundID",
            "SnapNum",
            "FileNr",
            "SubhaloIndex",
            "SubHalfMass",
        ],
        "formats": [
            "<i4",
            "<i4",
            "<i4",
            "<i4",
            "<i4",
            "<i4",
            "<f4",
            "<f4",
            "<f4",
            ("<f4", (3,)),
            ("<f4", (3,)),
            "<f4",
            "<f4",
            ("<f4", (3,)),
            "<i8",
            "<i4",
            "<i4",
            "<i4",
            "<f4",
        ],
        "offsets": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 48, 60, 64, 68, 80, 88, 92, 96, 100],
        "itemsize": 104,
    }
)


@dataclass(frozen=True)
class LHaloPartition:
    """Random-access view of one L-Halo partition and its per-tree counts."""

    path: Path
    tree_halo_counts: np.ndarray
    tree_first_halo: np.ndarray
    total_halos: int
    data_offset: int

    @property
    def tree_count(self) -> int:
        return int(self.tree_halo_counts.size)

    def read_tree(self, tree_index: int, *, copy: bool = True) -> np.ndarray:
        """Read one merger tree without materializing the rest of the partition."""

        if not 0 <= tree_index < self.tree_count:
            raise IndexError(f"tree_index {tree_index} is outside [0, {self.tree_count})")
        count = int(self.tree_halo_counts[tree_index])
        first = int(self.tree_first_halo[tree_index])
        records = np.memmap(
            self.path,
            dtype=LHALO_DTYPE,
            mode="r",
            offset=self.data_offset + first * LHALO_DTYPE.itemsize,
            shape=(count,),
        )
        return np.array(records, copy=True) if copy else records


def open_lhalo_partition(path) -> LHaloPartition:
    """Read the host-endian L-Halo header and return a validated partition view."""

    path = Path(path)
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
        if header.size != 2:
            raise ValueError(f"{path} does not contain a complete L-Halo header")
        tree_count, total_halos = (int(value) for value in header)
        if tree_count < 0 or total_halos < 0:
            raise ValueError(
                f"{path} reports invalid counts: trees={tree_count}, halos={total_halos}"
            )
        counts = np.fromfile(stream, dtype="<i4", count=tree_count)
        if counts.size != tree_count or np.any(counts < 0):
            raise ValueError(f"{path} contains an invalid per-tree halo-count table")
        data_offset = stream.tell()

    counted_halos = int(counts.astype(np.int64).sum())
    if counted_halos != total_halos:
        raise ValueError(
            f"{path} header reports {total_halos} halos but tree counts sum to {counted_halos}"
        )
    expected_size = data_offset + total_halos * LHALO_DTYPE.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path} has {actual_size} bytes; its header and 104-byte records require "
            f"{expected_size}"
        )

    first = np.empty(tree_count, dtype=np.int64)
    if tree_count:
        first[0] = 0
        np.cumsum(counts[:-1], dtype=np.int64, out=first[1:])
    counts.setflags(write=False)
    first.setflags(write=False)
    return LHaloPartition(path, counts, first, total_halos, data_offset)
