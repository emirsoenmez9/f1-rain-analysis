"""1. adım: seçilen sezonlardaki tüm yarışlar için hava durumu, grid ve sonuç verisini çek.

Çıktılar:
    data/raw/races.csv    -> her satır bir yarış (hava özeti ile)
    data/raw/results.csv  -> her satır bir yarış x sürücü (grid, bitiş, sürücü adı)

Kullanım:
    python src/fetch_races.py
    python src/fetch_races.py --years 2023 2024 2025
"""
import argparse
import hashlib
import json
import sys
import time

import pandas as pd
import requests

from common import CACHE, RAW

BASE = "https://api.openf1.org/v1"


def get(endpoint, retries=5, **params):
    """OpenF1'e istek at. Aynı istek bir kez indirilir, sonra data/cache/'den okunur."""
    key = hashlib.md5(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
    cache_file = CACHE / f"{endpoint}_{key}.json"
    if cache_file.exists():
        return pd.DataFrame(json.loads(cache_file.read_text()))

    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            time.sleep(5 * (attempt + 1))   # bağlantı hatası: bekle, tekrar dene
            continue
        if r.status_code == 404:        # sonuç yok
            return pd.DataFrame()
        if r.status_code == 429 or r.status_code >= 500:   # istek sınırı / sunucu hatası
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
        time.sleep(0.4)
        return pd.DataFrame(data)
    raise RuntimeError(f"{endpoint}: istek başarısız oldu (bağlantı ya da istek sınırı), biraz sonra tekrar dene.")


def to_dt(s):
    try:
        return pd.to_datetime(s, utc=True, format="ISO8601")
    except ValueError:
        return pd.to_datetime(s, utc=True)


def summarize_weather(w, start, end):
    """Yarış süresince hava özeti: yağmurlu ölçüm oranı, sıcaklıklar, nem."""
    empty = dict(n_weather=0, rain_frac=None, track_temp=None, air_temp=None, humidity=None)
    if w.empty or "rainfall" not in w:
        return empty
    w = w.copy()
    w["date"] = to_dt(w["date"])
    w = w[(w["date"] >= start) & (w["date"] <= end)]
    if w.empty:
        return empty
    return dict(
        n_weather=len(w),
        rain_frac=float((w["rainfall"].astype(float) > 0).mean()),
        track_temp=float(w["track_temperature"].mean()),
        air_temp=float(w["air_temperature"].mean()),
        humidity=float(w["humidity"].mean()),
    )


def build_results(result, grid, drivers):
    res = result.rename(columns={"position": "finish_position"})
    if not grid.empty:
        g = grid[["driver_number", "position"]].rename(columns={"position": "grid_position"})
        res = res.merge(g, on="driver_number", how="left")
    else:
        res["grid_position"] = None
    if not drivers.empty:
        d = drivers[["driver_number", "name_acronym", "full_name", "team_name"]].drop_duplicates("driver_number")
        res = res.merge(d, on="driver_number", how="left")
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025],
                        help="Sezonlar (varsayılan: 2023 2024 2025)")
    args = parser.parse_args()

    race_rows, result_frames, skipped = [], [], []

    for year in args.years:
        sessions = get("sessions", year=year, session_name="Race")
        print(f"{year}: {len(sessions)} yarış bulundu")
        for _, s in sessions.iterrows():
            sk = int(s["session_key"])
            name = f"{year} {s.get('circuit_short_name', s.get('country_name'))}"
            start, end = to_dt(pd.Series([s["date_start"]]))[0], to_dt(pd.Series([s["date_end"]]))[0]

            weather = get("weather", session_key=sk)
            result = get("session_result", session_key=sk)
            if result.empty:
                skipped.append(name)
                print("  atlandı (sonuç yok):", name)
                continue
            grid = get("starting_grid", session_key=sk)
            drivers = get("drivers", session_key=sk)

            race_rows.append(dict(
                session_key=sk, year=year,
                country_name=s.get("country_name"), circuit=s.get("circuit_short_name"),
                date_start=s["date_start"], **summarize_weather(weather, start, end),
            ))
            res = build_results(result, grid, drivers)
            res["session_key"] = sk
            res["year"] = year
            res = res[["session_key", "year"] + [c for c in res.columns if c not in ("session_key", "year")]]
            result_frames.append(res)
            print("  tamam:", name)

    if not race_rows:
        sys.exit("Hiç yarış verisi çekilemedi.")

    races = pd.DataFrame(race_rows)
    results = pd.concat(result_frames, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    races.to_csv(RAW / "races.csv", index=False)
    results.to_csv(RAW / "results.csv", index=False)

    print(f"\nYarış: {len(races)} | Sonuç satırı: {len(results)} | Atlanan: {len(skipped)}")
    rainy = races[races["rain_frac"].fillna(0) > 0].sort_values("rain_frac", ascending=False)
    print(f"\nYağmur görülen yarışlar ({len(rainy)}):")
    print(rainy[["year", "circuit", "rain_frac"]].round(3).to_string(index=False))
    print("\nKaydedildi:", RAW / "races.csv", "|", RAW / "results.csv")


if __name__ == "__main__":
    main()
