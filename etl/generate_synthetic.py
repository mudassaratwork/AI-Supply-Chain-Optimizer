"""Synthetic SKU-day sales generator.

Writes ``data/raw/sales.csv`` with daily sales for N SKUs over a date range.
Each SKU has its own baseline, linear trend, weekly seasonality (weekend
spike), yearly seasonality (December bump), random promo bursts, and
lognormal multiplicative noise.

Data-generating process (per SKU `i`, per day `t`):

    demand_i(t) = mu_i
                * (1 + alpha_i * progress(t))         # trend
                * (1 + beta_i  * is_weekend(t))       # weekly
                * (1 + gamma_i * dec_bump(t))         # yearly
                * promo_i(t)                          # random burst
                * lognormal(0, sigma_i)               # noise
    units_i(t)  = max(0, round(demand_i(t)))

`progress` runs 0 -> 1 across the range; `dec_bump` is a Gaussian on the
day-of-year centred near Dec 16.

CLI
---
    python -m etl.generate_synthetic                  # 3y x 50 SKUs -> data/raw/sales.csv
    python -m etl.generate_synthetic --num-skus 5 --years 1

Programmatic
------------
    from etl.generate_synthetic import generate_sales
    df = generate_sales(start="2022-01-01", end="2022-01-31", num_skus=3, seed=0)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer

DEFAULT_OUTPUT = Path("data/raw/sales.csv")
DEFAULT_NUM_SKUS = 50
DEFAULT_YEARS = 3
DEFAULT_SEED = 42

SALES_COLUMNS: tuple[str, ...] = ("date", "sku", "units_sold", "unit_price", "is_promo")


def _sku_id(i: int) -> str:
    return f"SKU-{i + 1:05d}"


def generate_sales(
    *,
    start: str | date,
    end: str | date,
    num_skus: int = DEFAULT_NUM_SKUS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return a SKU-day sales DataFrame.

    Columns: ``date`` (datetime64[ns]), ``sku`` (str), ``units_sold`` (int64),
    ``unit_price`` (float64), ``is_promo`` (bool).
    """
    if num_skus < 1:
        raise ValueError("num_skus must be >= 1")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    if len(dates) == 0:
        raise ValueError("date range is empty (start must be <= end)")
    n_days = len(dates)

    # Per-SKU parameters, drawn once so repeated runs with the same seed match.
    baselines = rng.uniform(5.0, 200.0, num_skus)
    trends = rng.uniform(-0.15, 0.30, num_skus)
    weekly_amp = rng.uniform(0.10, 0.50, num_skus)
    yearly_amp = rng.uniform(0.20, 0.80, num_skus)
    noise_sigma = rng.uniform(0.10, 0.25, num_skus)
    promo_prob = rng.uniform(0.01, 0.04, num_skus)
    promo_mult = rng.uniform(1.5, 3.0, num_skus)
    prices = rng.uniform(5.0, 200.0, num_skus).round(2)

    progress = np.arange(n_days, dtype=float) / max(n_days - 1, 1)
    # pandas 2.x: dayofweek / dayofyear may already be ndarrays, so go through
    # np.asarray to stay tolerant of the underlying return type.
    is_weekend = np.asarray(dates.dayofweek >= 5, dtype=float)
    day_of_year = np.asarray(dates.dayofyear, dtype=float)
    # Gaussian bump centred on day 350 (~Dec 16), sigma ~ 25 days.
    dec_bump = np.exp(-((day_of_year - 350.0) ** 2) / (2.0 * 25.0**2))

    frames: list[pd.DataFrame] = []
    for i in range(num_skus):
        trend = 1.0 + trends[i] * progress
        weekly = 1.0 + weekly_amp[i] * is_weekend
        yearly = 1.0 + yearly_amp[i] * dec_bump
        promo_days = rng.random(n_days) < promo_prob[i]
        promo = np.where(promo_days, promo_mult[i], 1.0)
        noise = rng.lognormal(mean=0.0, sigma=noise_sigma[i], size=n_days)

        demand = baselines[i] * trend * weekly * yearly * promo * noise
        units = np.clip(np.round(demand), 0, None).astype(np.int64)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku": _sku_id(i),
                    "units_sold": units,
                    "unit_price": prices[i],
                    "is_promo": promo_days,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    # Lock column order so downstream readers don't depend on dict insertion.
    return df[list(SALES_COLUMNS)]


app = typer.Typer(add_completion=False, help="Synthetic sales CSV generator.")


@app.callback(invoke_without_command=True)
def main(
    start: str | None = typer.Option(
        None, "--start", help="ISO start date (YYYY-MM-DD). Defaults to --years before --end."
    ),
    end: str | None = typer.Option(
        None, "--end", help="ISO end date (YYYY-MM-DD). Defaults to today."
    ),
    years: int = typer.Option(
        DEFAULT_YEARS, "--years", min=1, help="Span when --start is omitted."
    ),
    num_skus: int = typer.Option(DEFAULT_NUM_SKUS, "--num-skus", min=1),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "--output", help="Output CSV path."),
) -> None:
    """Generate the synthetic sales CSV and write it to ``--output``."""
    end_date = date.fromisoformat(end) if end else date.today()
    start_date = (
        date.fromisoformat(start) if start else end_date - timedelta(days=365 * years)
    )

    df = generate_sales(start=start_date, end=end_date, num_skus=num_skus, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    typer.echo(
        f"Wrote {len(df):,} rows ({num_skus} SKUs x "
        f"{(end_date - start_date).days + 1} days) to {output}"
    )


if __name__ == "__main__":
    app()
