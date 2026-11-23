"""6b. adım: H3 düzeltilmiş analiz — attrition ve ortalamaya gerileme artefaktını temizle.

İki bağımsız test:
  A) Yarış içi bitiş yüzdeliği (0=son bitiren, 100=1. bitiren) üzerinde:
         finish_pct ~ grid_pct * is_wet_int + C(team_year)
     Artık = gerçek_yüzdelik - beklenen_yüzdelik (pozitif = beklenenden iyi).
  B) Aynı yarışta ikisi de bitiren takım arkadaşına karşı sıra farkı (tm_margin):
     Doğrudan ıslak vs kuru permütasyon testi + Benjamini-Hochberg (FDR).

Çıktılar:
    results/h3_driver_rain_clean.csv
    results/h3_driver_rain_clean.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from common import RAW

PROCESSED = RAW.parent / "processed"
RESULTS = RAW.parent.parent / "results"


def perm_diff_mean(wet: np.ndarray, dry: np.ndarray, n_perm: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    obs = float(np.mean(wet) - np.mean(dry))
    pooled = np.concatenate([wet, dry])
    n_wet = len(wet)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(float(np.mean(perm[:n_wet]) - np.mean(perm[n_wet:]))) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def compute_teammate_margin(df: pd.DataFrame) -> pd.Series:
    margin = pd.Series(np.nan, index=df.index, dtype=float)
    for (_, _), g in df.groupby(["session_key", "team_name"]):
        if len(g) == 2:
            idx = g.index.to_list()
            pos = g["finish_position"].to_list()
            margin.loc[idx[0]] = pos[1] - pos[0]
            margin.loc[idx[1]] = pos[0] - pos[1]
    return margin


def make_plot(res_a: pd.DataFrame, res_b: pd.DataFrame, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5.5, 0.38 * len(res_a))))

    # Panel A: Yüzdelik artık
    df_a = res_a.sort_values("fark_pct", ascending=True)
    cols_a = ["#2a9d8f" if v > 0 else "#e76f51" for v in df_a["fark_pct"]]
    ax1.barh(df_a["pilot"], df_a["fark_pct"], color=cols_a, alpha=0.85)
    ax1.axvline(0, color="black", lw=1)
    for i, (_, r) in enumerate(df_a.iterrows()):
        off = 0.6 if r["fark_pct"] >= 0 else -0.6
        ha = "left" if r["fark_pct"] >= 0 else "right"
        ax1.text(r["fark_pct"] + off, i, f"p={r['p_ham']:.3f} | q={r['q_fdr']:.3f}", va="center", ha=ha, fontsize=7.5)
    ax1.set_title("A) Düzeltilmiş Yüzdelik Artık\n(Attrition + Islak Grid Eğimi Kontrollü)")
    ax1.set_xlabel("Islak − Kuru Fark (Yüzdelik Puan, + = Daha İyi)")
    ax1.set_xlim(df_a["fark_pct"].min() - 14, df_a["fark_pct"].max() + 16)
    ax1.grid(axis="x", alpha=0.25)

    # Panel B: Takım arkadaşı farkı
    df_b = res_b.sort_values("tm_fark", ascending=True)
    cols_b = ["#2a9d8f" if v > 0 else "#e76f51" for v in df_b["tm_fark"]]
    ax2.barh(df_b["pilot"], df_b["tm_fark"], color=cols_b, alpha=0.85)
    ax2.axvline(0, color="black", lw=1)
    for i, (_, r) in enumerate(df_b.iterrows()):
        off = 0.2 if r["tm_fark"] >= 0 else -0.2
        ha = "left" if r["tm_fark"] >= 0 else "right"
        ax2.text(r["tm_fark"] + off, i, f"p={r['p_ham']:.3f} | q={r['q_fdr']:.3f} (n={int(r['n_islak'])})", va="center", ha=ha, fontsize=7.5)
    ax2.set_title("B) Takım Arkadaşına Karşı Sıra Farkı\n(Aynı Araç & Aynı Yarış, İkisi de Bitirenler)")
    ax2.set_xlabel("Islak − Kuru Fark (Sıra, + = Takım Arkadaşına Karşı Daha İyi)")
    ax2.set_xlim(df_b["tm_fark"].min() - 4.2, df_b["tm_fark"].max() + 4.8)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("2023–2025 F1 Pilotları: Artefakttan Arındırılmış H3 Yağmur Testleri", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    dr = pd.read_csv(PROCESSED / "driver_races.csv")
    fin = dr[
        dr["classified"]
        & dr["grid_position"].notna()
        & dr["finish_position"].notna()
        & dr["team_name"].notna()
        & dr["name_acronym"].notna()
    ].copy()

    # Yarış içi yüzdelik skor (100 = 1. bitiren, 0 = son bitiren)
    fin["n_fin_race"] = fin.groupby("session_key")["finish_position"].transform("count")
    fin["fin_rank"] = fin.groupby("session_key")["finish_position"].rank(method="average")
    fin["finish_pct"] = 100.0 * (1.0 - (fin["fin_rank"] - 1.0) / (fin["n_fin_race"] - 1.0))

    fin["n_grid_race"] = fin.groupby("session_key")["grid_position"].transform("count")
    fin["grid_rank"] = fin.groupby("session_key")["grid_position"].rank(method="average")
    fin["grid_pct"] = 100.0 * (1.0 - (fin["grid_rank"] - 1.0) / (fin["n_grid_race"] - 1.0))

    fin["is_wet_int"] = fin["is_wet_tyre"].astype(int)
    fin["team_year"] = fin["team_name"].astype(str) + "_" + fin["year"].astype(str)

    # Model A: Attrition + ıslak grid eğimi düzleşmesi kontrollü
    m = smf.ols("finish_pct ~ grid_pct * is_wet_int + C(team_year)", data=fin).fit()
    fin["perf_pct"] = fin["finish_pct"] - m.fittedvalues

    print(f"=== MODEL A ÖZETİ (R² = {m.rsquared:.3f}, n = {len(fin)}) ===")
    for term in ("grid_pct", "is_wet_int", "grid_pct:is_wet_int"):
        print(f"  {term:22s}: katsayı = {m.params[term]:+.4f} (p = {m.pvalues[term]:.4f})")

    rows_a = []
    for pilot, g in fin.groupby("name_acronym"):
        wet = g.loc[g["is_wet_tyre"], "perf_pct"].to_numpy()
        dry = g.loc[~g["is_wet_tyre"], "perf_pct"].to_numpy()
        if len(wet) < 4 or len(dry) < 15:
            continue
        diff, p_raw = perm_diff_mean(wet, dry, n_perm=10_000, seed=100 + len(rows_a))
        rows_a.append({
            "pilot": pilot,
            "n_islak": len(wet),
            "n_kuru": len(dry),
            "islak_ort": round(float(np.mean(wet)), 2),
            "kuru_ort": round(float(np.mean(dry)), 2),
            "fark_pct": round(diff, 2),
            "p_ham": round(p_raw, 4),
        })
    res_a = pd.DataFrame(rows_a).sort_values("fark_pct", ascending=False).reset_index(drop=True)
    res_a["q_fdr"] = np.round(multipletests(res_a["p_ham"], method="fdr_bh")[1], 4)

    # Model B: Takım arkadaşı farkı permütasyon + FDR
    fin["tm_margin"] = compute_teammate_margin(fin)
    rows_b = []
    for pilot, g in fin.dropna(subset=["tm_margin"]).groupby("name_acronym"):
        wet = g.loc[g["is_wet_tyre"], "tm_margin"].to_numpy()
        dry = g.loc[~g["is_wet_tyre"], "tm_margin"].to_numpy()
        if len(wet) < 3 or len(dry) < 15:
            continue
        diff, p_raw = perm_diff_mean(wet, dry, n_perm=10_000, seed=200 + len(rows_b))
        rows_b.append({
            "pilot": pilot,
            "n_islak": len(wet),
            "n_kuru": len(dry),
            "tm_islak": round(float(np.mean(wet)), 2),
            "tm_kuru": round(float(np.mean(dry)), 2),
            "tm_fark": round(diff, 2),
            "p_ham": round(p_raw, 4),
        })
    res_b = pd.DataFrame(rows_b).sort_values("tm_fark", ascending=False).reset_index(drop=True)
    res_b["q_fdr"] = np.round(multipletests(res_b["p_ham"], method="fdr_bh")[1], 4)

    RESULTS.mkdir(parents=True, exist_ok=True)
    merged = res_a.merge(
        res_b[["pilot", "n_islak", "tm_fark", "p_ham", "q_fdr"]].rename(
            columns={"n_islak": "tm_n_islak", "p_ham": "tm_p_ham", "q_fdr": "tm_q_fdr"}
        ),
        on="pilot",
        how="outer",
    )
    merged.to_csv(RESULTS / "h3_driver_rain_clean.csv", index=False)
    make_plot(res_a, res_b, RESULTS / "h3_driver_rain_clean.png")

    print("\n=== TEST A: DÜZELTİLMİŞ YÜZDELİK ARTIK (Attrition + Islak Grid Eğimi Kontrollü) ===")
    print(res_a.to_string(index=False))
    print(f"Ham p < 0,05: {int((res_a['p_ham'] < 0.05).sum())}/{len(res_a)} | FDR q < 0,05: {int((res_a['q_fdr'] < 0.05).sum())}/{len(res_a)}")

    print("\n=== TEST B: TAKIM ARKADAŞI FARKI (En az 3 ıslak & 15 kuru çiftli bitiriş) ===")
    print(res_b.to_string(index=False))
    print(f"Ham p < 0,05: {int((res_b['p_ham'] < 0.05).sum())}/{len(res_b)} | FDR q < 0,05: {int((res_b['q_fdr'] < 0.05).sum())}/{len(res_b)}")
    print(f"\nKaydedildi: {RESULTS / 'h3_driver_rain_clean.csv'} | {RESULTS / 'h3_driver_rain_clean.png'}")


if __name__ == "__main__":
    main()
