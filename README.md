# AnalizaFUT

Predyktor wyniku meczu reprezentacji (W/D/L) i prognoza Mistrzostw Świata 2026.
Model uczony na ~150 latach historii meczów reprezentacji.

## Co to robi

- **Predyktor 1×2** dla dowolnej pary reprezentacji — `P(home win)`, `P(draw)`, `P(away win)` z modelu Gradient Boosting.
- **Symulacja Monte Carlo** WC 2026 (10 000 turniejów) — szanse mistrzostwa, finału, półfinału, ćwierćfinału i awansu z grupy dla każdej z 48 drużyn.
- **Streamlit UI** z trzema zakładkami: Predyktor, Dashboard WC 2026, Model (collinearity, importance, współczynniki LR).
- **Raport markdown** — `reports/wc2026_report.md` z wykresami w `reports/figures/`.

## Stack

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `streamlit`. Python 3.11+.

## Dane

[Kaggle — *International football results from 1872 to 2017* (martj42)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/data).
Pobierz CSV-e (`results.csv`, `goalscorers.csv`, `shootouts.csv`, `former_names.csv`)
i wrzuć do `data/`.

## Cechy modelu

- **Elo rating** każdej drużyny (start 1500, K-factor 60/50/40/20 zależnie od rangi turnieju, mnożnik różnicy bramek FIFA-style).
- **Forma** — średnie punkty/gole strzelone/stracone z 10 ostatnich meczów.
- **Head-to-head** — średnie punkty z 5 ostatnich bezpośrednich starć.
- **Tier turnieju** (one-hot), boisko neutralne, gospodarz, dni od ostatniego meczu.
- **Mapowanie historycznych nazw** (ZSRR→Rosja, Zair→DR Konga itd.) z zachowaniem ciągłości Elo.

## Walidacja

Time-based split: trening do 2022-12-31, hold-out 2023-01-01 → 2026-03-31 (3 445 meczów, w tym Euro 2024 i kwalifikacje WC 2026).

| Model | Accuracy | Log-loss |
|---|---|---|
| Baseline (always H) | 46.85% | — |
| Logistic Regression | 60.09% | 0.866 |
| Gradient Boosting | **60.52%** | **0.865** |
| Random Forest | 60.35% | 0.868 |

## Uruchomienie

```bash
pip install -r requirements.txt

# trening modelu (~30 s)
python src/model.py

# symulacja Monte Carlo WC 2026 (~10 s)
python src/simulate_wc.py

# generacja raportu MD + wykresów PNG
python src/report_gen.py

# UI
python -m streamlit run app/streamlit_app.py
```

## Struktura

```
data/                        — surowe CSV z Kaggle
src/
  data_loader.py             — wczytanie + mapowanie historycznych nazw
  features.py                — Elo, forma, H2H, tier (chronologiczny walk)
  model.py                   — trening 3 modeli, walidacja, zapis pkl
  simulate_wc.py             — auto-detekcja grup + Monte Carlo turnieju
  report_gen.py              — figury PNG + raport MD
app/
  streamlit_app.py           — UI
models/
  predictor.pkl              — wytrenowany pipeline (gen. przez model.py)
reports/
  wc2026_report.md           — raport
  figures/                   — wykresy PNG
  simulation_results.csv     — wyniki Monte Carlo
  match_predictions.csv      — predykcje 72 meczów grupowych
  groups.csv                 — przypisanie drużyn do grup
```

## Top faworyci wg modelu

1. Hiszpania — 18.5%
2. Argentyna — 13.4%
3. Francja — 10.1%
4. Anglia — 7.4%
5. Brazylia — 4.7%

Pełna tabela: [`reports/wc2026_report.md`](reports/wc2026_report.md).

## Ograniczenia

- Brak składów osobowych, kontuzji, xG — predykcja na poziomie zespołu.
- Drabinka pucharowa losowana w każdej iteracji Monte Carlo (oficjalny mapping bracket→grupy nie jest publikowany).
- Tiebreaker grupowy = Elo (różnicy bramek nie symulujemy explicite).
- Klasa "draw" jest niedopredykowana — typowe dla 3-klasowej klasyfikacji wyniku meczu. Patrz na rozkład `P`, nie argmax.
