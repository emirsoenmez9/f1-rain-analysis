"""3. adım: Gerçek ıslak yarışları OpenF1 stints (INTERMEDIATE / WET lastik) verisiyle doğrula.

Checkpoint'lidir: bağlantı koparsa aynı komutu tekrar çalıştır, kaldığı yerden devam eder.
Çıktı: data/raw/wet_tyres.csv
"""
import sys
import time

import pandas as pd

from common import RAW
from fetch_races import get


def main():
    races_path = RAW / "races.csv"
    out_path = RAW / "wet_tyres.csv"
    if not races_path.exists():
        sys.exit("races.csv yok. Önce: python src/fetch_races.py")

    races = pd.read_csv(races_path)

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

        st = get("stints", session_key=sk)
        if st.empty:
            empty.append(f"{r['year']} {r['circuit']}")
            print("  boş:", r["year"], r["circuit"])
            continue

        comp = st["compound"].fillna("").astype(str).str.upper()
        lap_start = pd.to_numeric(st["lap_start"], errors="coerce")
        lap_end = pd.to_numeric(st["lap_end"], errors="coerce")
        laps = (lap_end - lap_start + 1).clip(lower=0).fillna(0)

        is_inter = comp == "INTERMEDIATE"
        is_wet = comp == "WET"
        is_rain_tyre = is_inter | is_wet

        total_laps = float(laps.sum())
        wet_laps = float(laps[is_rain_tyre].sum())

        row = pd.DataFrame([{
            "session_key": sk,
            "n_stints": len(st),
            "inter_stints": int(is_inter.sum()),
            "wet_stints": int(is_wet.sum()),
            "wet_tyre_drivers": int(st.loc[is_rain_tyre, "driver_number"].nunique()),
            "wet_tyre_laps": int(wet_laps),
            "total_stint_laps": int(total_laps),
            "wet_lap_frac": round(wet_laps / total_laps, 4) if total_laps > 0 else 0.0,
        }])

        write_header = not out_path.exists() or out_path.stat().st_size == 0
        row.to_csv(out_path, mode="a", header=write_header, index=False)
        done.add(sk)
        print("  kaydedildi:", r["year"], r["circuit"], f"(yağmur lastiği tur oranı: {row['wet_lap_frac'].iloc[0]:.2f})")
        time.sleep(0.3)

    if not out_path.exists():
        sys.exit("Hiç stint verisi çekilemedi.")

    wt = pd.read_csv(out_path).drop_duplicates(subset=["session_key"])
    wt.to_csv(out_path, index=False)

    m = races.merge(wt, on="session_key", how="left")
    ilgili = m[(m["rain_frac"].fillna(0) > 0) | (m["wet_tyre_drivers"].fillna(0) > 0)].sort_values(
        "wet_lap_frac", ascending=False
    )

    print(f"\nStint çekilen yarış: {wt.session_key.nunique()} / {len(races)} | boş: {len(empty)}")
    print("\nYağmur sensörü VEYA yağmur lastiği görülen tüm yarışlar:")
    cols = ["year", "circuit", "rain_frac", "wet_tyre_drivers", "inter_stints", "wet_stints", "wet_lap_frac"]
    print(ilgili[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
