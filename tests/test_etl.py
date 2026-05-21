"""Tests for ``etl.generate_synthetic``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from etl.generate_synthetic import SALES_COLUMNS, app, generate_sales

# ---------------------------------------------------------------------------
# generate_sales (pure function)
# ---------------------------------------------------------------------------


def test_generate_sales_shape() -> None:
    df = generate_sales(start="2022-01-01", end="2022-01-14", num_skus=3, seed=0)
    # 14 inclusive days x 3 SKUs.
    assert len(df) == 14 * 3


def test_generate_sales_columns_and_order() -> None:
    df = generate_sales(start="2022-01-01", end="2022-01-07", num_skus=2, seed=0)
    assert tuple(df.columns) == SALES_COLUMNS


def test_generate_sales_no_nulls() -> None:
    df = generate_sales(start="2022-01-01", end="2022-01-07", num_skus=2, seed=0)
    assert not df.isna().any().any()


def test_generate_sales_units_non_negative_integer() -> None:
    df = generate_sales(start="2022-01-01", end="2022-01-31", num_skus=5, seed=0)
    assert (df["units_sold"] >= 0).all()
    assert pd.api.types.is_integer_dtype(df["units_sold"])


def test_generate_sales_price_positive() -> None:
    df = generate_sales(start="2022-01-01", end="2022-01-07", num_skus=4, seed=0)
    assert (df["unit_price"] > 0).all()


def test_generate_sales_distinct_skus() -> None:
    df = generate_sales(start="2022-01-01", end="2022-01-07", num_skus=4, seed=0)
    assert df["sku"].nunique() == 4
    assert set(df["sku"]) == {"SKU-00001", "SKU-00002", "SKU-00003", "SKU-00004"}


def test_generate_sales_deterministic() -> None:
    a = generate_sales(start="2022-01-01", end="2022-01-31", num_skus=5, seed=42)
    b = generate_sales(start="2022-01-01", end="2022-01-31", num_skus=5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_generate_sales_empty_range_rejected() -> None:
    with pytest.raises(ValueError):
        generate_sales(start="2022-01-10", end="2022-01-01", num_skus=2, seed=0)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_writes_csv(tmp_path: Path) -> None:
    """`python -m etl.generate_synthetic` end-to-end: writes a valid CSV."""
    output = tmp_path / "sales.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--start",
            "2022-01-01",
            "--end",
            "2022-01-07",
            "--num-skus",
            "2",
            "--seed",
            "0",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()

    df = pd.read_csv(output)
    assert len(df) == 7 * 2
    assert tuple(df.columns) == SALES_COLUMNS
    assert (df["units_sold"] >= 0).all()
