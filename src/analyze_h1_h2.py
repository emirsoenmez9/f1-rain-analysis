"""5. adım: H1 (sıralama bozulması) ve H2 (DNF riski) hipotez testleri.

- Yarış düzeyinde (n = 70) 10.000 tekrarlı permütasyon testi (ortalama & medyan farkı) + Mann-Whitney U
- 3 farklı ıslak yarış tanımıyla duyarlılık analizi + sürekli doz-tepki (wet_lap_frac)
- Sürücü x yarış düzeyinde DNF lojistik regresyonu: klasik SE vs yarışa göre kümelenmiş SE

Çıktılar:
    results/h1_h2_summary.csv
    results/h1_h2_wet_vs_dry.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from common import RAW

PROCESSED = RAW.parent / "processed"
RESULTS = RAW.parent.parent / "results"


def perm_test(wet: np.ndarray, dry: np.ndarray, stat_fn, n_perm: int = 10_000, seed: int = 42):
    """İki yönlü kesin Monte Carlo permütasyon testi."""
    rng = np.random.default_rng(seed)
    obs = float(stat_fn(wet) - stat_fn(dry))
    pooled = np.concatenate([wet, dry])
    n_wet = len(wet)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(stat_fn(perm[:n_wet]) - stat_fn(perm[n_wet:]))
        if abs(diff) >= abs(obs):
            count += 1
    p_val = (count + 1) / (n_perm + 1)
    return obs, p_val


def run_h1_h2_race_level(rs: pd.DataFrame) -> pd.DataFrame:
    rs = rs.copy()
    rs["is_wet_broad"] = rs["wet_tyre_drivers"].fillna(0) >= 10

    defs = [
        ("lastik_5pct (n=7)", "is_wet_tyre"),
        ("lastik_10drv (n=8)", "is_wet_broad"),
        ("sensor_10pct (n=9)", "is_wet_sensor"),
    ]
    metrics = [
        ("spearman_grid_finish", "Spearman (grid-bitiş)"),
        ("mean_abs_shift", "Ort. mutlak sıra kayması"),
        ("dnf_rate", "DNF oranı"),
    ]

    rows = []
    for def_name, flag_col in defs:
        for col, label in metrics:
            wet = rs.loc[rs[flag_col], col].dropna().to_numpy()
            dry = rs.loc[~rs[flag_col], col].dropna().to_numpy()

            diff_mean, p_perm_mean = perm_test(wet, dry, np.mean)
            diff_med, p_perm_med = perm_test(wet, dry, np.median)
            _, p_mwu = stats.mannwhitneyu(wet, dry, alternative="two-sided")

            rows.append({
                "tanim": def_name,
                "metrik": label,
                "n_islak": len(wet),
                "n_kuru": len(dry),
                "islak_ort": round(float(np.mean(wet)), 4),
                "kuru_ort": round(float(np.mean(dry)), 4),
                "ort_fark": round(diff_mean, 4),
                "p_perm_ort": round(p_perm_mean, 4),
                "islak_med": round(float(np.median(wet)), 4),
                "kuru_med": round(float(np.median(dry)), 4),
                "med_fark": round(diff_med, 4),
                "p_perm_med": round(p_perm_med, 4),
                "p_mwu": round(float(p_mwu), 4),
            })
    return pd.DataFrame(rows)


def run_dose_response(rs: pd.DataFrame):
    print("\n=== SÜREKLİ DOZ-TEPKİ (Eşik bağımsız: wet_lap_frac ile Spearman) ===")
    for col, label in [
        ("spearman_grid_finish", "Spearman (grid-bitiş)"),
        ("mean_abs_shift", "Ort. mutlak sıra kayması"),
        ("dnf_rate", "DNF oranı"),
    ]:
        sub = rs[["wet_lap_frac", col]].dropna()
        rho, p = stats.spearmanr(sub["wet_lap_frac"], sub[col])
        print(f"  {label:26s}: rho = {rho:+.3f}, p = {p:.4f} (n = {len(sub)})")


def run_dnf_logit(dr: pd.DataFrame):
    print("\n=== H2: DNF LOJİSTİK REGRESYONU (Start alanlar | Klasik vs Yarışa Göre Kümelenmiş SE) ===")
    starters = dr[dr["started"]].copy()
    starters["dnf_int"] = starters["dnf"].astype(int)
    starters["is_wet_broad"] = starters["wet_tyre_drivers"].fillna(0) >= 10

    for label, col in [
        ("lastik_5pct (7 yarış)", "is_wet_tyre"),
        ("lastik_10drv (8 yarış)", "is_wet_broad"),
        ("sensor_10pct (9 yarış)", "is_wet_sensor"),
    ]:
        sub = starters[["dnf_int", col, "session_key"]].dropna().copy()
        sub["wet_int"] = sub[col].astype(int)

        m_cls = smf.logit("dnf_int ~ wet_int", data=sub).fit(disp=False)
        m_clu = smf.logit("dnf_int ~ wet_int", data=sub).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": sub["session_key"]},
        )

        coef = float(m_clu.params["wet_int"])
        odds_ratio = float(np.exp(coef))
        ci_lo, ci_hi = np.exp(m_clu.conf_int().loc["wet_int"])
        print(
            f"  {label:24s} | OR = {odds_ratio:.2f} (%95 GA kümelenmiş: [{ci_lo:.2f}, {ci_hi:.2f}])"
            f" | SE klasik={m_cls.bse['wet_int']:.3f} (p={m_cls.pvalues['wet_int']:.4f})"
            f" vs kümelenmiş={m_clu.bse['wet_int']:.3f} (p={m_clu.pvalues['wet_int']:.4f})"
        )


def make_plot(rs: pd.DataFrame, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    specs = [
        ("spearman_grid_finish", "Grid–Bitiş Spearman Korelasyonu", "Düşük = daha fazla kaos"),
        ("mean_abs_shift", "Ortalama Mutlak Sıra Kayması", "Yüksek = daha fazla kaos"),
        ("dnf_rate", "Yarış Dışı Kalma (DNF) Oranı", "Start alanlara göre"),
    ]
    rng = np.random.default_rng(7)

    for ax, (col, title, subtitle) in zip(axes, specs):
        dry = rs.loc[~rs["is_wet_tyre"], col].dropna().to_numpy()
        wet = rs.loc[rs["is_wet_tyre"], col].dropna().to_numpy()

        ax.boxplot(
            [dry, wet],
            tick_labels=[f"Kuru\n(n={len(dry)})", f"Islak\n(n={len(wet)})"],
            widths=0.45,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.8},
        )
        ax.scatter(
            1 + rng.normal(0, 0.06, size=len(dry)),
            dry,
            alpha=0.5,
            s=22,
            color="#4c78a8",
        )
        ax.scatter(
            2 + rng.normal(0, 0.05, size=len(wet)),
            wet,
            alpha=0.85,
            s=36,
            color="#e45756",
        )
        ax.set_title(f"{title}\n({subtitle})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("2023–2025 F1 Yarışları: Islak (Lastik ≥ %5 Tur) vs. Kuru", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    rs = pd.read_csv(PROCESSED / "race_summary.csv")
    dr = pd.read_csv(PROCESSED / "driver_races.csv")

    summary = run_h1_h2_race_level(rs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "h1_h2_summary.csv", index=False)

    print("=== H1 & H2: YARIŞ DÜZEYİNDE PERMÜTASYON & MANN-WHITNEY TESTLERİ ===")
    print(summary.to_string(index=False))

    run_dose_response(rs)
    run_dnf_logit(dr)

    fig_path = RESULTS / "h1_h2_wet_vs_dry.png"
    make_plot(rs, fig_path)
    print(f"\nKaydedildi: {RESULTS / 'h1_h2_summary.csv'} | {fig_path}")


if __name__ == "__main__":
    main()
