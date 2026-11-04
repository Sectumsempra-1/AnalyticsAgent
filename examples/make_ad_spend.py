"""Build examples/ad_spend_sales.csv: a "does X cause Y?" demo with a hidden confounder.

Peak season (Ramadan/Eid weeks and November-December) raises both ad spend (the
marketing team spends more) and sales (customers buy more). Within a season, ad spend
has only a small effect on sales. Run: python examples/make_ad_spend.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "ad_spend_sales.csv"
# Approximate Ramadan + Eid al-Fitr weeks
PEAK_RANGES = [("2023-03-20", "2023-04-25"), ("2024-03-08", "2024-04-14"), ("2025-02-26", "2025-04-03")]


def is_peak(day: pd.Timestamp) -> bool:
    in_holiday = any(pd.Timestamp(a) <= day <= pd.Timestamp(b) for a, b in PEAK_RANGES)
    return in_holiday or day.month in (11, 12)


def main() -> None:
    rng = np.random.default_rng(42)
    weeks = pd.date_range("2023-01-02", "2025-12-29", freq="W-MON")
    rows = []
    for week in weeks:
        peak = is_peak(week)
        ad_spend = rng.normal(26_000 if peak else 11_000, 3_000)
        demand = rng.normal(310_000 if peak else 180_000, 22_000)  # driven by the season
        sales = demand + 0.4 * ad_spend  # small true ad effect
        rows.append({
            "week_start": week.date().isoformat(),
            "season": "peak" if peak else "normal",
            "ad_spend": round(ad_spend, 2),
            "sales": round(sales, 2),
        })  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} weeks to {OUT}")
    print("overall corr:", round(df["ad_spend"].corr(df["sales"]), 3))
    for season, g in df.groupby("season"):
        print(f"{season}: n={len(g)} corr={g['ad_spend'].corr(g['sales']):.3f}")


if __name__ == "__main__":
    main()
