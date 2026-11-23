"""2. adım: starting_grid boş geldiği için başlangıç sırasını position verisinden türet.

Her yarışı çektikçe diske yazar (checkpoint). Bağlantı koparsa aynı komutu
tekrar çalıştır; zaten inmiş yarışları atlayıp kaldığı yerden devam eder.
"""
import sys
import time

import pandas as pd

from common import RAW
from fetch_races import get, to_dt


def main():
    races_path = RAW / "races.csv"
    results_path = RAW / "results.csv"
    out_path = RAW / "grid_from_position.csv"

    if not races_path.exists() or not results_path.exists():
        sys.exit("races.csv / results.csv yok. Önce: python src/fetch_races.py")

    races = pd.read_csv(races_path)
    results = pd.read_csv(results_path)

    # Checkpoint: daha önce çekilen yarışları oku
    if out_path.exists() and out_path.stat().st_size > 0:
        done = set(pd.read_csv(out_path)["session_key"].unique())
        print(f"Önbellekten devam ediliyor: {len(done)} yarış zaten çekilmiş.")
    else:
        done = set()

    empty = []
    for _, r in races.iterrows():
        sk = int(r["session_key"])
        if sk in done:
            continue

        pos = get("position", session_key=sk)
        if pos.empty:
            empty.append(f"{r['year']} {r['circuit']}")
            print("  boş:", r["year"], r["circuit"])
            continue

        pos = pos.copy()
        pos["date"] = to_dt(pos["date"])
        first = (
            pos.sort_values("date")
            .groupby("driver_number", as_index=False)
            .first()[["driver_number", "position", "date"]]
            .rename(columns={"position": "grid_position", "date": "grid_time"})
        )
        first["session_key"] = sk
        first = first[["session_key", "driver_number", "grid_position", "grid_time"]]

        # Yarış biter bitmez diske yaz
        write_header = not out_path.exists() or out_path.stat().st_size == 0
        first.to_csv(out_path, mode="a", header=write_header, index=False)
        done.add(sk)
        print("  kaydedildi:", r["year"], r["circuit"], f"({len(first)} sürücü)")
        time.sleep(0.3)

    if not out_path.exists():
        sys.exit("Hiç position verisi çekilemedi.")

    # ---- Kendini kontrol ----
    grid = pd.read_csv(out_path).drop_duplicates(subset=["session_key", "driver_number"])
    grid.to_csv(out_path, index=False)
    grid["grid_time"] = pd.to_datetime(grid["grid_time"], utc=True, format="ISO8601")

    print(f"\nGrid türetilen yarış: {grid.session_key.nunique()} / {len(races)} | boş: {len(empty)}")

    m = results[["session_key", "driver_number", "finish_position"]].merge(
        grid, on=["session_key", "driver_number"], how="inner"
    )
    m_valid = m.dropna(subset=["finish_position", "grid_position"])
    corr = m_valid.groupby("session_key").apply(
        lambda g: g["grid_position"].corr(g["finish_position"], method="spearman"),
        include_groups=False,
    )
    print("\nYarış başına grid-bitiş Spearman korelasyonu:")
    print(corr.describe().round(3).to_string())
    print("Korelasyonu 0,99 üstü olan (şüpheli) yarış sayısı:", int((corr > 0.99).sum()))

    rs = races.set_index("session_key")["date_start"]
    dt = grid.groupby("session_key")["grid_time"].min()
    fark = (
        dt - pd.to_datetime(rs.loc[dt.index], utc=True, format="ISO8601")
    ).dt.total_seconds() / 60
    print(
        "\nİlk kayıt, yarış başlangıcına göre (dk) -- medyan:",
        round(float(fark.median()), 1),
        "| min:",
        round(float(fark.min()), 1),
        "| max:",
        round(float(fark.max()), 1),
    )

    # Göz kontrolü: 2024 Interlagos (Verstappen #1, P17 civarı başlayıp P1 bitirmeli)
    bra = races[
        (races["year"] == 2024) & (races["circuit"].str.contains("Interlagos", na=False))
    ]
    if not bra.empty:
        sk_bra = int(bra.iloc[0]["session_key"])
        sub = m[m["session_key"] == sk_bra].sort_values("finish_position").head(5)
        print("\nGöz kontrolü -- 2024 Interlagos ilk 5:")
        print(sub[["driver_number", "grid_position", "finish_position"]].to_string(index=False))

    print("\nKaydedildi:", out_path)


if __name__ == "__main__":
    main()

