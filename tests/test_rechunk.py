import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import rechunk_parquet  # noqa: E402


def test_rechunk_preserves_rows_and_order(tmp_path):
    path = str(tmp_path / "meta.parquet")
    df = pd.DataFrame({"doi": [f"10.1/{i}" for i in range(10_000)], "abstract": ["x" * 50] * 10_000})
    df.to_parquet(path, row_group_size=10_000)
    assert rechunk_parquet.needs_rechunk(path, 1000)

    rechunk_parquet.rechunk(path, 1000)

    assert pq.ParquetFile(path).metadata.num_row_groups == 10
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)
    assert not rechunk_parquet.needs_rechunk(path, 1000)
    assert [p for p in os.listdir(tmp_path) if "tmp" in p] == []
