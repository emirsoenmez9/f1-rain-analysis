[Türkçe Sürüm (README.tr.md)](README.tr.md)

---

# f1-rain-analysis

Statistical analysis of how rain affects finishing-order volatility, DNF risk, and driver performance across 2023–2025 Formula 1 races (`n = 70` races, `1,397` driver-race observations via the OpenF1 REST API).

## Research Questions & Findings

### 1. Defining a Wet Race (Weather Sensor vs. On-Track Reality)
Relying solely on weather sensors (`rain_frac`) creates false positives—for instance, 2023 Spa recorded `rain_frac = 0.110` while zero drivers fitted wet tyres, and at 2025 Spa all 20 drivers ran `INTERMEDIATE` tyres only on Lap 1 (`97.7%` of race laps were dry). Therefore, the primary wet-race definition uses on-track `INTERMEDIATE`/`WET` tyre share (`wet_lap_frac >= 0.05`, `n = 7` wet races), with sensitivity checks across `n = 8` (including 2025 Spa) and `n = 9` (sensor `>= 0.10`).

### 2. H1 — Does Rain Disrupt the Finishing Order?
- Wet races show lower mean grid–finish Spearman correlation (`0.644` vs. `0.771`; 10,000-permutation `p = 0.064`, Mann–Whitney `p = 0.223`) and larger mean absolute position shift (`3.54` vs. `2.79`; `p = 0.057`).
- Although the median difference is sharp (`p = 0.005`), the distribution is **bimodal**: 4 of the 7 wet races produced heavy chaos (`0.35–0.56`), whereas 3 finished even more orderly (`0.85–0.93`) than the dry median (`0.823`).
- **Verdict:** The direction consistently favors greater chaos, but rain does not disrupt races uniformly; with `n = 7`, the mean effect remains borderline at the `5%` level.

### 3. H2 — Does Rain Increase Attrition (DNF) Risk?
- The primary in-race DNF rate is numerically higher in wet races (`16.25%` vs. `10.96%`), but not statistically significant in race-level permutation tests (`p = 0.110`; `p = 0.322` when 2025 Spa is included).
- In driver-level logistic regression, race-clustered SE (`0.234`, `p = 0.049`) was smaller than classical SE (`0.250`, `p = 0.066`) and jumped to `p = 0.319` after adding a single race—a textbook case of **few treated clusters bias** (`7` wet clusters).
- **Exploratory Sensitivity:** Including rain-related pre-race/in-race `DNS` and `DSQ` exits yields a significantly higher total non-finish rate in wet races (`20.0%` vs. `11.9%`, `p_perm_mean = 0.021`, `p_mwu = 0.026`).

### 4. H3 — Are "Rain Master" Drivers Real?
- An initial OLS model (`finish_position ~ grid_position + C(team_year)`) flagged `6/17` drivers at `FDR q < 0.05`, but diagnostics revealed this was partly driven by higher wet-race attrition and flattened wet grid slopes inflating backmarker residuals.
- After correcting for both artifacts via **Test A** (`finish_pct ~ grid_pct * is_wet + C(team_year)`) and cross-checking with **Test B** (same-race teammate finishing margin), **zero positive "rain masters" passed `FDR q < 0.05` across both tests (`0/17`)**: `GAS` passed only Test A (`q = 0.007`, Test B `q = 0.974`), while `VER` passed only Test B (`+5.36`, `q = 0.040`—largely a zero-sum mirror of teammate `PER`'s collapse).
- In contrast, **two negative outliers passed FDR in both tests**: `LEC` (Test A `q = 0.007`, Test B `-5.02`, `q = 0.040`) and `PER` (Test A `q = 0.010`, Test B `-7.57`, `q = 0.022`). Rather than creating statistical "rain superheroes," wet conditions disproportionately penalize error-prone driver/pit-wall pairings.

## Reproduction

    pip install -r requirements.txt
    python src/fetch_races.py
    python src/fetch_grid.py
    python src/fetch_wet_tyres.py
    python src/build_dataset.py
    python src/analyze_h1_h2.py
    python src/diagnose.py
    python src/analyze_h3_v2.py

## Limitations

- Low statistical power due to only `7–8` genuinely wet races across three seasons.
- Red flags and safety-car starts censor extreme wet-weather crash risk.
- Circuit characteristics and safety-car pit timing are not modeled as covariates.
