"""4. adım: Ham OpenF1 tablolarını birleştirip temiz analiz tablolarını üret.

Çıktılar (data/processed/):
    driver_races.csv -> sürücü x yarış düzeyinde (H2 ve H3 için)
    race_summary.csv -> yarış düzeyinde özet (H1 ve H2 için, n = 70)
"""
import sys

import numpy as np
import pandas as pd

from common import RAW

PROCESSED = RAW.parent / "processed"


def to_bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def main():
    required = ["races.csv", "results.csv", "grid_from_position.csv", "wet_tyres.csv"]
    for name in required:
        if not (RAW / name).exists():
            sys.exit(f"Eksik dosya: data/raw/{name}")

    races = pd.read_csv(RAW / "races.csv")
    results = pd.read_csv(RAW / "results.csv").drop(columns=["grid_position"], errors="ignore")
    grid = pd.read_csv(RAW / "grid_from_position.csv")[
        ["session_key", "driver_number", "grid_position"]
    ]
    wet = pd.read_csv(RAW / "wet_tyres.csv")

    if wet["session_key"].nunique() < len(races):
        sys.exit(
            f"wet_tyres.csv eksik ({wet['session_key'].nunique()}/{len(races)}). "
            "Önce tekrar çalıştır: python src/fetch_wet_tyres.py"
        )

    # Yarış düzeyinde hava + lastik birleşimi
    race_meta = races.merge(wet, on="session_key", how="left")
    race_meta["is_wet_tyre"] = race_meta["wet_lap_frac"].fillna(0) >= 0.05
    race_meta["is_wet_sensor"] = race_meta["rain_frac"].fillna(0) >= 0.10

    # Sürücü x yarış tablosu
    df = results.merge(grid, on=["session_key", "driver_number"], how="left")
    df = df.merge(
        race_meta[
            [
                "session_key",
                "circuit",
                "country_name",
                "rain_frac",
                "wet_lap_frac",
                "wet_tyre_drivers",
                "is_wet_tyre",
                "is_wet_sensor",
                "track_temp",
                "air_temp",
                "humidity",
            ]
        ],
        on="session_key",
        how="left",
    )

    for col in ("dnf", "dns", "dsq"):
        df[col] = to_bool(df[col])

    df["started"] = ~df["dns"]
    df["classified"] = (~df["dnf"]) & (~df["dns"]) & (~df["dsq"]) & df["finish_position"].notna()
    df["pos_gain"] = np.where(
        df["classified"] & df["grid_position"].notna(),
        df["grid_position"] - df["finish_position"],
        np.nan,
    )
    df["abs_pos_shift"] = df["pos_gain"].abs()

    # Yarış düzeyinde metrikler
    rows = []
    for sk, g in df.groupby("session_key"):
        starters = g[g["started"]]
        fin = g[g["classified"] & g["grid_position"].notna()]
        spearman = (
            fin["grid_position"].corr(fin["finish_position"], method="spearman")
            if len(fin) >= 5
            else np.nan
        )
        rows.append(
            {
                "session_key": sk,
                "n_entries": len(g),
                "n_starters": int(len(starters)),
                "n_finishers": int(len(fin)),
                "n_dnf": int(starters["dnf"].sum()),
                "n_dns": int(g["dns"].sum()),
                "n_dsq": int(g["dsq"].sum()),
                "dnf_rate": round(starters["dnf"].sum() / len(starters), 4)
                if len(starters) > 0
                else np.nan,
                "spearman_grid_finish": round(float(spearman), 4) if pd.notna(spearman) else np.nan,
                "mean_abs_shift": round(float(fin["abs_pos_shift"].mean()), 4)
                if len(fin) > 0
                else np.nan,
            }
        )

    race_summary = race_meta.merge(pd.DataFrame(rows), on="session_key", how="left")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "driver_races.csv", index=False)
    race_summary.to_csv(PROCESSED / "race_summary.csv", index=False)

    print("=== GENEL KONTROL ===")
    print(f"Yarış: {len(race_summary)} | Sürücü-yarış satırı: {len(df)}")
    print(
        f"DNF: {int(df['dnf'].sum())} | DNS: {int(df['dns'].sum())} | "
        f"DSQ: {int(df['dsq'].sum())} | Grid eksik (start alanlarda): "
        f"{int((df['started'] & df['grid_position'].isna()).sum())}"
    )

    print("\n=== ISLAK YARIŞ KARŞILAŞTIRMASI (sensör vs lastik) ===")
    ilgili = race_summary[
        (race_summary["rain_frac"].fillna(0) > 0)
        | (race_summary["wet_tyre_drivers"].fillna(0) > 0)
    ].sort_values("wet_lap_frac", ascending=False)
    cols = [
        "year",
        "circuit",
        "rain_frac",
        "wet_lap_frac",
        "is_wet_tyre",
        "is_wet_sensor",
        "n_dnf",
        "spearman_grid_finish",
        "mean_abs_shift",
    ]
    print(ilgili[cols].round(3).to_string(index=False))

    print("\n=== HIZLI ÖN BAKIŞ: ISLAK (lastik) vs KURU ===")
    print(
        race_summary.groupby("is_wet_tyre")[
            ["spearman_grid_finish", "mean_abs_shift", "dnf_rate"]
        ]
        .agg(["count", "mean", "median"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
