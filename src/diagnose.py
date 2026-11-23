"""Ara teşhisler: 2025 Spa stint kontrolü, uç kuru yarış ve DNS+DSQ dahil H2 duyarlılığı."""
import numpy as np
import pandas as pd
from scipy import stats

from common import RAW
from fetch_races import get

PROCESSED = RAW.parent / "processed"


def perm_test(wet: np.ndarray, dry: np.ndarray, stat_fn, n_perm: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    obs = float(stat_fn(wet) - stat_fn(dry))
    pooled = np.concatenate([wet, dry])
    n_wet = len(wet)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(float(stat_fn(perm[:n_wet]) - stat_fn(perm[n_wet:]))) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    rs = pd.read_csv(PROCESSED / "race_summary.csv")

    print("=== 1) 2025 SPA STINT KONTROLÜ (session_key=9939) ===")
    st = get("stints", session_key=9939)
    if st.empty:
        print("Stint verisi boş döndü.")
    else:
        st["laps"] = st["lap_end"] - st["lap_start"] + 1
        print(
            st.groupby("compound")[["lap_start", "lap_end", "laps"]]
            .agg(["count", "min", "max", "mean"])
            .round(2)
            .to_string()
        )

    print("\n=== 2) KURU YARIŞLARDAKİ EN EKSTREM 3 KAOS YARIŞI ===")
    dry_races = rs[~rs["is_wet_tyre"]].copy()
    cols = ["year", "circuit", "n_dnf", "dnf_rate", "spearman_grid_finish", "mean_abs_shift"]
    print(dry_races.sort_values("spearman_grid_finish").head(3)[cols].round(3).to_string(index=False))

    print("\n=== 3) H2 DUYARLILIK: TOPLAM BİTİREMEME ORANI (DNF + DNS + DSQ) ===")
    rs["non_fin_rate"] = (rs["n_dnf"] + rs["n_dns"] + rs["n_dsq"]) / rs["n_entries"]
    rs["is_wet_broad"] = rs["wet_tyre_drivers"].fillna(0) >= 10

    print("Islak yarışlarda DNF / DNS / DSQ kırılımı:")
    wcols = ["year", "circuit", "n_dnf", "n_dns", "n_dsq", "dnf_rate", "non_fin_rate"]
    print(rs[rs["is_wet_broad"]][wcols].round(3).to_string(index=False))

    print("\nPermütasyon & Mann-Whitney testleri (non_fin_rate):")
    for label, col in [("lastik_5pct (n=7)", "is_wet_tyre"), ("lastik_10drv (n=8)", "is_wet_broad")]:
        wet = rs.loc[rs[col], "non_fin_rate"].dropna().to_numpy()
        dry = rs.loc[~rs[col], "non_fin_rate"].dropna().to_numpy()
        d_mean, p_mean = perm_test(wet, dry, np.mean)
        d_med, p_med = perm_test(wet, dry, np.median)
        _, p_mwu = stats.mannwhitneyu(wet, dry, alternative="two-sided")
        print(
            f"  {label:20s} | islak_ort={wet.mean():.3f} vs kuru_ort={dry.mean():.3f} "
            f"(fark={d_mean:+.3f}, p_perm_ort={p_mean:.4f}) | "
            f"p_perm_med={p_med:.4f} | p_mwu={p_mwu:.4f}"
        )


if __name__ == "__main__":
    main()
