from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_data
from features import build_features

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def _plt_setup():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig_top_favourites(sim, out, n=12):
    top = sim.nlargest(n, "p_champion")[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["team"], top["p_champion"] * 100, color="#1f77b4")
    ax.set_xlabel("Champion probability (%)")
    ax.set_title(f"Top {n} favourites — FIFA World Cup 2026 (Monte Carlo, N=10 000)")
    for y, v in zip(top["team"], top["p_champion"] * 100):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_stage_progression(sim, out, n=10):
    top = sim.nlargest(n, "p_champion")
    stages = ["p_advance", "p_qf", "p_semi", "p_final", "p_champion"]
    labels = ["Group adv.", "Quarter", "Semi", "Final", "Champion"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    for _, row in top.iterrows():
        ax.plot(x, [row[s] * 100 for s in stages], marker="o", label=row["team"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Tournament-stage probabilities — top 10 favourites")
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_group_heatmap(groups_df, out):
    pivot_rows = []
    for g, sub in groups_df.groupby("group"):
        sub = sub.sort_values("p_advance", ascending=False).reset_index(drop=True)
        for rank, row in sub.iterrows():
            pivot_rows.append({"group": g, "rank": rank + 1,
                               "team": row["team"], "p_advance": row["p_advance"]})
    pv = pd.DataFrame(pivot_rows)
    mat = pv.pivot(index="group", columns="rank", values="p_advance")
    labels = pv.pivot(index="group", columns="rank", values="team")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat.values * 100, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"#{i}" for i in mat.columns])
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"Group {g}" for g in mat.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            t = labels.values[i, j]
            v = mat.values[i, j] * 100
            ax.text(j, i, f"{t}\n{v:.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(advance from group), %")
    ax.set_title("Group-stage qualification probabilities")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_recent_form(played, teams, out, months=12):
    cutoff = played["date"].max() - pd.DateOffset(months=months)
    rec = played[played["date"] >= cutoff].copy()
    rows = []
    for team in teams:
        m_h = rec[rec["home_team"] == team]
        m_a = rec[rec["away_team"] == team]
        n = len(m_h) + len(m_a)
        if n == 0:
            continue
        pts = ((m_h["outcome"] == "H").sum() * 3 + (m_h["outcome"] == "D").sum() * 1
               + (m_a["outcome"] == "A").sum() * 3 + (m_a["outcome"] == "D").sum() * 1)
        gf = m_h["home_score"].sum() + m_a["away_score"].sum()
        ga = m_h["away_score"].sum() + m_a["home_score"].sum()
        rows.append({"team": team, "matches": n, "pts_per_match": pts / n,
                     "gf_per_match": gf / n, "ga_per_match": ga / n})
    form = pd.DataFrame(rows).sort_values("pts_per_match", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(form["team"], form["pts_per_match"], color="#2ca02c")
    ax.set_xlabel(f"Points per match (last {months} months)")
    ax.set_title(f"Recent form of key teams (last {months} months)")
    for y, v, n in zip(form["team"], form["pts_per_match"], form["matches"]):
        ax.text(v + 0.02, y, f"{v:.2f} ({n} m)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return form


def top_scorers(goalscorers, teams, months=24, top_n=3):
    cutoff = goalscorers["date"].max() - pd.DateOffset(months=months)
    rec = goalscorers[(goalscorers["date"] >= cutoff)
                      & (~goalscorers["own_goal"].astype(bool))]
    out = {}
    for t in teams:
        sub = rec[rec["team"] == t]
        if sub.empty:
            out[t] = []
            continue
        cnt = sub["scorer"].value_counts().head(top_n)
        out[t] = list(cnt.items())
    return out


def write_report(sim, groups_df, matches, played, form_df, scorers, model_meta):
    md = []
    md.append("# Mistrzostwa Świata 2026 — analiza i prognoza")
    md.append("\n## 1. Wstęp i metodologia\n")
    md.append("Turniej **FIFA World Cup 2026** rozegrany zostanie w **USA, Meksyku i Kanadzie** "
              "od **11.06.2026 do 19.07.2026**. Po raz pierwszy w historii w finałach wystąpi "
              "**48 reprezentacji** podzielonych na 12 grup po 4 zespoły. Polska nie zakwalifikowała "
              "się do turnieju.")
    md.append("\n### Dane")
    md.append("- Źródło: [Kaggle — *International football results from 1872 to 2017* (martj42)]"
              "(https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/data)")
    md.append("- 49 215 rozegranych meczów reprezentacji (1872 → 2026-03-31)")
    md.append("- 47 601 strzelców goli, 675 serii rzutów karnych, 36 mapowań historycznych nazw")
    md.append("- 72 nierozegranych meczów fazy grupowej WC 2026 jako zbiór do predykcji")
    md.append("\n### Stack")
    md.append("`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `streamlit`. Python 3.14.")
    md.append("\n### Cechy modelu (per mecz, stan przed meczem)")
    md.append("- Elo rating każdej drużyny — startowy 1500, K-factor 60 (WC) / 50 (kont.) / "
              "40 (eliminacje) / 20 (sparingi), z modyfikatorem różnicy bramek (FIFA-style).")
    md.append("- Forma — średnie punkty/gole strzelone/stracone z 10 ostatnich meczów.")
    md.append("- Head-to-head — średnie punkty z 5 ostatnich bezpośrednich starć.")
    md.append("- Tier turnieju (one-hot), boisko neutralne, gospodarz, dni od ostatniego meczu.")
    md.append("\n### Model i walidacja")
    md.append(f"- Porównanie 3 modeli (Logistic Regression / Gradient Boosting / Random Forest) "
              f"na podziale czasowym: trening do 2022-12-31, walidacja 2023-01-01 → 2026-03-31.")
    md.append(f"- Wybrany model: **{model_meta['best_model']}** — "
              f"accuracy **{model_meta['test_metrics']['accuracy']*100:.2f}%** "
              f"vs baseline (zawsze gospodarz) **46.85%**, "
              f"log-loss **{model_meta['test_metrics']['log_loss']:.3f}**.")
    md.append("- Symulacja całego turnieju (group + knockout): Monte Carlo, **10 000 powtórzeń**.")

    md.append("\n## 2. Top faworyci do tytułu\n")
    md.append("![Top favourites](figures/top_favourites.png)\n")
    md.append("| # | Reprezentacja | P(mistrz) | P(finał) | P(półfinał) | P(ćwierćfinał) | P(awans z grupy) |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    top = sim.nlargest(15, "p_champion").reset_index(drop=True)
    for i, r in top.iterrows():
        md.append(f"| {i+1} | **{r['team']}** | {r['p_champion']*100:.2f}% "
                  f"| {r['p_final']*100:.2f}% | {r['p_semi']*100:.2f}% "
                  f"| {r['p_qf']*100:.2f}% | {r['p_advance']*100:.2f}% |")

    md.append("\n## 3. Predykcja podium\n")
    podium = sim.nlargest(3, "p_champion").reset_index(drop=True)
    medals = ["Złoto", "Srebro", "Brąz"]
    for i, r in podium.iterrows():
        md.append(f"- **{medals[i]}** — {r['team']} (P_champion = {r['p_champion']*100:.1f}%)")

    md.append("\n## 4. Typowani ćwierćfinaliści (P_QF ≥ 25%)\n")
    qf_list = sim[sim["p_qf"] >= 0.25].sort_values("p_qf", ascending=False)
    md.append("| Reprezentacja | P(ćwierćfinał) | P(półfinał) | Elo |")
    md.append("|---|---:|---:|---:|")
    for _, r in qf_list.iterrows():
        md.append(f"| {r['team']} | {r['p_qf']*100:.1f}% | {r['p_semi']*100:.1f}% | {r['elo']:.0f} |")
    md.append("\n![Stage progression](figures/stage_progression.png)")

    md.append("\n## 5. Forma kluczowych drużyn (ostatnie 12 miesięcy)\n")
    md.append("![Form](figures/recent_form.png)\n")
    md.append("| Drużyna | Mecze | Pkt/mecz | GF/mecz | GA/mecz |")
    md.append("|---|---:|---:|---:|---:|")
    for _, r in form_df.sort_values("pts_per_match", ascending=False).iterrows():
        md.append(f"| {r['team']} | {r['matches']} | {r['pts_per_match']:.2f} "
                  f"| {r['gf_per_match']:.2f} | {r['ga_per_match']:.2f} |")
    md.append("\n### Top strzelcy w ostatnich 24 miesiącach\n")
    for team, lst in scorers.items():
        if not lst:
            md.append(f"- **{team}** — brak danych")
            continue
        s = ", ".join(f"{name} ({n})" for name, n in lst)
        md.append(f"- **{team}** — {s}")

    md.append("\n## 6. Predykcje wszystkich 72 meczów fazy grupowej\n")
    md.append("| Data | Mecz | P(1) | P(X) | P(2) | Typ |")
    md.append("|---|---|---:|---:|---:|---|")
    for _, r in matches.iterrows():
        probs = {"H": r["P_home_win"], "D": r["P_draw"], "A": r["P_away_win"]}
        pick = max(probs, key=probs.get)
        pick_label = "1" if pick == "H" else ("X" if pick == "D" else "2")
        md.append(f"| {r['date']} | {r['home_team']} – {r['away_team']} "
                  f"| {r['P_home_win']*100:.1f}% | {r['P_draw']*100:.1f}% "
                  f"| {r['P_away_win']*100:.1f}% | **{pick_label}** |")

    md.append("\n## 7. Analiza grup\n")
    md.append("![Groups heatmap](figures/groups_heatmap.png)\n")
    for g, sub in groups_df.groupby("group"):
        sub = sub.sort_values("p_advance", ascending=False)
        md.append(f"\n**Grupa {g}**")
        for _, r in sub.iterrows():
            md.append(f"- {r['team']} — P(awans) = {r['p_advance']*100:.1f}% (Elo {r['elo']:.0f})")

    md.append("\n## 8. Ograniczenia modelu\n")
    md.append("- Brak składów osobowych — model nie wie o kontuzjach. Predykcja bazuje wyłącznie "
              "na formie zespołowej i historii.")
    md.append("- Brak xG, posiadania, statystyk klubowych — dataset zawiera tylko wyniki meczów.")
    md.append("- Drabinka pucharowa losowana — oficjalnego mapowania bracket-grup nie znamy, "
              "więc w każdej iteracji Monte Carlo pary R32 są losowane. Topowe drużyny czasem "
              "spotykają się wcześniej niż w realnym losowaniu.")
    md.append("- Tiebreaker grupowy = Elo — różnicy bramek nie symulujemy explicite.")

    text = "\n".join(md)
    out = REPORTS / "wc2026_report.md"
    out.write_text(text, encoding="utf-8")
    print(f"Report saved -> {out} ({len(text)} chars)")


def main():
    _plt_setup()
    d = load_data()
    played, fixtures, _ = build_features(d["played"], d["fixtures"])
    model = joblib.load(ROOT / "models" / "predictor.pkl")

    sim = pd.read_csv(REPORTS / "simulation_results.csv")
    matches = pd.read_csv(REPORTS / "match_predictions.csv")
    groups_df = pd.read_csv(REPORTS / "groups.csv")

    fig_top_favourites(sim, FIGURES / "top_favourites.png")
    fig_stage_progression(sim, FIGURES / "stage_progression.png")
    fig_group_heatmap(groups_df, FIGURES / "groups_heatmap.png")

    key_teams = ["Spain", "Argentina", "France", "England", "Brazil",
                 "Germany", "Netherlands", "Portugal", "Morocco", "Japan"]
    form_df = fig_recent_form(d["played"], key_teams, FIGURES / "recent_form.png")
    scorers = top_scorers(d["goalscorers"], key_teams)

    write_report(sim, groups_df, matches, d["played"], form_df, scorers, {
        "best_model": model["best_model"],
        "test_metrics": model["test_metrics"],
    })


if __name__ == "__main__":
    main()
