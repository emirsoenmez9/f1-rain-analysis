[English Version (README.md)](README.md)

---

# f1-rain-analysis

2023–2025 Formula 1 yarış verileri (OpenF1 REST API, `n = 70` yarış, `1.397` sürücü–yarış gözlemi) üzerinden yağmurun sıralama bozulmasına, yarış dışı kalma (DNF) riskine ve pilot performansına etkisini inceleyen istatistiksel analiz projesi.

## Araştırma Soruları ve Bulgular

### 1. Islak Yarış Tanımı (Sensör vs. Pist Gerçeği)
Hava durumu sensörüne (`rain_frac`) tek başına güvenmek yalancı pozitif üretmektedir (ör. 2023 Spa'da `rain_frac = 0.110` olmasına karşın hiçbir sürücü yağmur lastiği takmamıştır; 2025 Spa'da ise 20 sürücü yalnızca 1. turda `INTERMEDIATE` kullanmış, yarışın `%97.7`'si kuru lastikle koşulmuştur). Bu nedenle birincil tanım olarak `INTERMEDIATE`/`WET` lastikle dönülen tur oranı (`wet_lap_frac >= 0.05`, `n = 7` ıslak yarış) seçilmiş, tüm testler `n = 8` (2025 Spa dahil) ve `n = 9` (sensör `>= 0.10`) tanımlarıyla duyarlılık analizinden geçirilmiştir.

### 2. H1 — Yağmur Sıralamayı Bozuyor mu?
- Islak yarışlarda grid–bitiş Spearman korelasyonu ortalama daha düşük (`0.644` vs `0.771`, 10.000 permütasyon `p = 0.064`, Mann-Whitney `p = 0.223`), ortalama mutlak sıra kayması daha yüksektir (`3.54` vs `2.79`, `p = 0.057`).
- Medyan farkı sert çıksa da (`p = 0.005`) bunun nedeni **iki tepeli (bimodal)** dağılımdır: 7 ıslak yarışın 4'ü ağır kaos (`0.35–0.56`), 3'ü ise kuru yarış medyanından (`0.823`) bile daha düzenli (`0.85–0.93`) bitmiştir.
- **Karar:** Yön tutarlı biçimde kaos lehinedir ancak etki her ıslak yarışta homojen değildir; ortalama etki düşük örneklemle (`n = 7`) `%5` sınırında kalmaktadır.

### 3. H2 — Yağmurda Yarış Dışı Kalma Riski Artıyor mu?
- Birincil metrik olan yarış içi DNF oranı ıslakta daha yüksek (`%16.25` vs `%10.96`) ancak yarış düzeyi permütasyon testinde anlamlı değildir (`p = 0.110`; 2025 Spa eklenince `p = 0.322`).
- Sürücü düzeyindeki lojistik regresyonda kümelenmiş SE (`0.234`, `p = 0.049`) klasik SE'den (`0.250`, `p = 0.066`) küçük çıkmış, tek yarış eklendiğinde `p = 0.319`'a fırlamıştır; bu durum yalnızca 7 ıslak kümeden kaynaklanan **az-küme yanlılığına (few treated clusters bias)** işaret eder.
- **Keşifsel Duyarlılık:** Yağmur kaynaklı start öncesi ve yarış içi `DNS` ve `DSQ` vakaları eklendiğinde toplam bitirememe oranı ıslakta belirgin biçimde artmaktadır (`%20.0` vs `%11.9`, `p_perm_ort = 0.021`, `p_mwu = 0.026`).

### 4. H3 — "Yağmur Ustası" Pilotlar Gerçek mi?
- İlk OLS modeli (`finish_position ~ grid_position + C(team_year)`) 6 pilotu FDR `q < 0.05` altında anlamlı çıkarsa da, bu sonucun bir kısmının ıslakta artan yıpranma (attrition) ve düzleşen grid eğimi artefaktından kaynaklandığı teşhis edilmiştir.
- **Düzeltilmiş Test A** (`finish_pct ~ grid_pct * is_wet + C(team_year)`) ve **Test B** (aynı yarışta ikisi de bitiren takım arkadaşına karşı sıra farkı) birlikte uygulandığında, iki testin kesişiminde FDR `q < 0.05` geçen **hiçbir pozitif "yağmur ustası" çıkmamıştır (`0/17`)**: `GAS` yalnızca Test A'da (`q = 0.007`, Test B'de `q = 0.974`), `VER` yalnızca Test B'de (`+5.36`, `q = 0.040` — büyük ölçüde takım arkadaşı `PER`'in çöküşünün yansıması) anlamlıdır.
- Buna karşılık **iki testte birden FDR'dan geçen iki negatif vaka** vardır: `LEC` (`Test A q = 0.007`, `Test B -5.02, q = 0.040`) ve `PER` (`Test A q = 0.010`, `Test B -7.57, q = 0.022`). Veri, yağmurun "süper kahraman" yaratmasından çok hata/strateji maliyetini ağırlaştırarak belirli pilot ve garajları cezalandırdığını göstermektedir.

## Çalıştırma Sırası

    pip install -r requirements.txt
    python src/fetch_races.py
    python src/fetch_grid.py
    python src/fetch_wet_tyres.py
    python src/build_dataset.py
    python src/analyze_h1_h2.py
    python src/diagnose.py
    python src/analyze_h3_v2.py

## Kısıtlar

- 3 sezonda yalnızca 7–8 gerçek ıslak yarış bulunması istatistiksel gücü sınırlamaktadır.
- Kırmızı bayrak ve güvenlik aracı uygulamaları ağır yağmurdaki gerçek kaza riskini aşağı yönlü sansürlemektedir.
- Pist karakteristiği ve güvenlik aracı zamanlaması modellere kovaryat olarak dahil edilmemiştir.
