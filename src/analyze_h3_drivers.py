"""6. adım: H3 — "Yağmur ustası" pilotlar gerçekten var mı, yoksa gürültü mü?

Çıktılar:
    results/h3_driver_rain.csv
    results/h3_driver_rain.png
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
    """Aynı yarışta aynı takımdan tam 2 bitiren varsa: rakip_bitiş - kendi_bitiş."""
    margin = pd.Series(np.nan, index=df.index, dtype=float)
    for (_, _), g in df.groupby(["session_key", "team_name"]):
        if len(g) == 2:
            idx = g.index.to_list()
            pos = g["finish_position"].to_list()
            margin.loc[idx[0]] = pos[1] - pos[0]
            margin.loc[idx[1]] = pos[0] - pos[1]
    return margin


def make_plot(res: pd.DataFrame, out_path):
    plot_df = res.sort_values("fark_islak_kuru", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.38 * len(plot_df))))

    colors = ["#2a9d8f" if v > 0 else "#e76f51" for v in plot_df["fark_islak_kuru"]]
    ax.barh(plot_df["pilot"], plot_df["fark_islak_kuru"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1)

    for i, (_, r) in enumerate(plot_df.iterrows()):
        offset = 0.08 if r["fark_islak_kuru"] >= 0 else -0.08
        ha = "left" if r["fark_islak_kuru"] >= 0 else "right"
        label = f"p={r['p_ham']:.3f} | FDR q={r['q_fdr']:.3f} (n={int(r['n_islak'])})"
        ax.text(r["fark_islak_kuru"] + offset, i, label, va="center", ha=ha, fontsize=8)

    ax.set_xlabel("Islak − Kuru Performans Farkı (Grid ve Takım-Sezon Kontrollü, + = Yağmurda Daha İyi)")
    ax.set_title("2023–2025 F1 Pilotları: Yağmur Etkisi ve Çoklu Test Düzeltmesi (FDR)")
    ax.set_xlim(
        plot_df["fark_islak_kuru"].min() - 2.4,
        plot_df["fark_islak_kuru"].max() + 2.8,
    )
    ax.grid(axis="x", alpha=0.25)
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

    fin["team_year"] = fin["team_name"].astype(str) + "_" + fin["year"].astype(str)

    model = smf.ols("finish_position ~ grid_position + C(team_year)", data=fin).fit()
    fin["expected_finish"] = model.fittedvalues
    fin["perf_resid"] = fin["expected_finish"] - fin["finish_position"]
    fin["tm_margin"] = compute_teammate_margin(fin)

    print(f"Beklenen bitiş modeli R² = {model.rsquared:.3f} (n = {len(fin)} bitiriş)")

    rows = []
    for pilot, g in fin.groupby("name_acronym"):
        wet = g.loc[g["is_wet_tyre"], "perf_resid"].to_numpy()
        dry = g.loc[~g["is_wet_tyre"], "perf_resid"].to_numpy()
        if len(wet) < 4 or len(dry) < 15:
            continue

        diff, p_raw = perm_diff_mean(wet, dry, n_perm=10_000, seed=42 + len(rows))

        tm_wet = g.loc[g["is_wet_tyre"], "tm_margin"].dropna()
        tm_dry = g.loc[~g["is_wet_tyre"], "tm_margin"].dropna()
        tm_diff = float(tm_wet.mean() - tm_dry.mean()) if (len(tm_wet) >= 2 and len(tm_dry) >= 5) else np.nan

        rows.append({
            "pilot": pilot,
            "n_islak": len(wet),
            "n_kuru": len(dry),
            "islak_perf": round(float(np.mean(wet)), 3),
            "kuru_perf": round(float(np.mean(dry)), 3),
            "fark_islak_kuru": round(diff, 3),
            "p_ham": round(p_raw, 4),
            "tm_fark_islak_kuru": round(tm_diff, 3) if pd.notna(tm_diff) else np.nan,
        })

    res = pd.DataFrame(rows).sort_values("fark_islak_kuru", ascending=False).reset_index(drop=True)
    _, q_vals, _, _ = multipletests(res["p_ham"], method="fdr_bh")
    res["q_fdr"] = np.round(q_vals, 4)

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "h3_driver_rain.csv"
    fig_path = RESULTS / "h3_driver_rain.png"
    res.to_csv(csv_path, index=False)
    make_plot(res, fig_path)

    print("\n=== H3: PİLOT BAZINDA YAĞMUR PERFORMANSI (En az 4 ıslak & 15 kuru bitiriş) ===")
    print(res.to_string(index=False))
    print(f"\nHam p < 0,05 olan pilot sayısı : {int((res['p_ham'] < 0.05).sum())} / {len(res)}")
    print(f"FDR q < 0,05 olan pilot sayısı : {int((res['q_fdr'] < 0.05).sum())} / {len(res)}")
    print(f"\nKaydedildi: {csv_path} | {fig_path}")


if __name__ == "__main__":
    main()
