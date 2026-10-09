"""
Rewrites parquet files with small row groups so the search app can read a few
rows without decoding whole files. Row order and content are unchanged, so the
alignment with the embedding .npy files is preserved. Writes to a temp file,
verifies row count + a sample of rows, then atomically replaces the original.

Run where the data lives (lab desktop before sync, or the server in a quiet
hour — it is I/O-bound, single-threaded, and needs ~1 GB RAM for bioRxiv):

    python tools/rechunk_parquet.py --row-group-size 2000 FILE.parquet [FILE ...]
    python tools/rechunk_parquet.py --dry-run /path/*.parquet     # report only

Files already at or below the target row-group size are skipped.
"""
import argparse
import os
import sys

import pyarrow.parquet as pq


def needs_rechunk(path: str, target: int) -> bool:
    meta = pq.ParquetFile(path).metadata
    return meta.num_row_groups > 0 and max(meta.row_group(i).num_rows for i in range(meta.num_row_groups)) > target


def rechunk(path: str, target: int) -> None:
    table = pq.read_table(path)
    tmp = f"{path}.rechunk-tmp"
    pq.write_table(table, tmp, row_group_size=target, compression="snappy", write_page_index=True)
    check = pq.read_table(tmp)
    if check.num_rows != table.num_rows or not check.slice(0, 100).equals(table.slice(0, 100)) \
            or not check.slice(table.num_rows - 100).equals(table.slice(table.num_rows - 100)):
        os.remove(tmp)
        raise RuntimeError(f"verification failed for {path}; original left untouched")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--row-group-size", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = [f for f in args.files if needs_rechunk(f, args.row_group_size)]
    print(f"{len(todo)} of {len(args.files)} file(s) need rechunking")
    for i, f in enumerate(todo, 1):
        if args.dry_run:
            print(f"  would rechunk {f}")
            continue
        rechunk(f, args.row_group_size)
        print(f"  [{i}/{len(todo)}] {f}")


if __name__ == "__main__":
    sys.exit(main())
