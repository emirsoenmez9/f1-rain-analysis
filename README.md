# f1-rain-analysis

2023–2025 Formula 1 yarış verileri (OpenF1 API, `n = 70` yarış) üzerinden yağmurun sıralama bozulmasına, yarış dışı kalma (DNF) riskine ve pilot performansına etkisini inceleyen veri analizi projesi.

## Proje Durumu

- [x] **Veri Toplama:** 2023–2025 yarış, hava durumu, sonuç, grid (`position` başlangıç kayıtları, medyan starttan 54 dk önce) ve `stints` lastik verileri.
- [x] **Veri Doğrulama:** Hava sensörü (`rain_frac`) yerine pist gerçeği (`INTERMEDIATE`/`WET` lastik kullanım oranı `wet_lap_frac >= 0.05`, `n = 7` ıslak yarış) birincil tanım olarak belirlendi; 3 farklı eşikle duyarlılık analizi yapıldı.
- [x] **H1 & H2 Analizi:** Yarış düzeyinde 10.000 tekrarlı permütasyon testi, Mann-Whitney U, sürekli doz-tepki ve yarışa göre kümelenmiş DNF lojistik regresyonu.
- [ ] **H3 Analizi:** Pilot bazında "yağmur ustası" iddiasının grid + araç-sezon kontrolü ve FDR düzeltmesiyle test edilmesi.

## H1 & H2 Ön Bulguları (`n = 70` yarış)

- **H1 (Sıralama Bozulması):** Islak yarışlarda grid–bitiş Spearman korelasyonu ortalama daha düşük (`0.644` vs `0.771`, `p_perm_ort = 0.064`, `p_mwu = 0.223`) ve ortalama mutlak sıra kayması daha yüksek (`3.54` vs `2.79`, `p_perm_ort = 0.057`). Medyan farkı sert çıksa da (`p = 0.005`) dağılım iki tepeli: 7 ıslak yarışın 4'ü yüksek kaos, 3'ü kuru yarış düzeninde bitiyor.
- **H2 (DNF Riski):** Islak yarışlarda DNF oranı sayısal olarak yüksek (`%16.2` vs `%11.0`) ancak yarış düzeyinde permütasyon testinde anlamlı değil (`p = 0.110`; 2025 Spa eklenince `p = 0.322`). Lojistik regresyondaki `p = 0.049` sonucu yalnızca 7 ıslak kümeden kaynaklanan az-küme yanlılığına duyarlı.

## Çalıştırma

    python src/fetch_races.py
    python src/fetch_grid.py
    python src/fetch_wet_tyres.py
    python src/build_dataset.py
    python src/analyze_h1_h2.py
