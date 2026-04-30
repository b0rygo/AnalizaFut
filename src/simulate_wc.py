from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_loader import load_data
from features import build_features

ROOT = Path(__file__).resolve().parents[1]
N_SIMS_DEFAULT = 10_000
RNG = np.random.default_rng(42)


def detect_groups(fixtures):
    adj = defaultdict(set)
    for _, r in fixtures.iterrows():
        adj[r["home_team"]].add(r["away_team"])
        adj[r["away_team"]].add(r["home_team"])

    seen = set()
    groups = []
    for team in adj:
        if team in seen:
            continue
        comp = set()
        stack = [team]
        while stack:
            t = stack.pop()
            if t in comp:
                continue
            comp.add(t)
            stack.extend(adj[t] - comp)
        seen |= comp
        if len(comp) != 4:
            raise ValueError(f"Group component size {len(comp)} != 4: {comp}")
        groups.append(sorted(comp))
    if len(groups) != 12:
        raise ValueError(f"Expected 12 groups, got {len(groups)}")
    return groups


def predict_group_matches(model, fixtures):
    proba = model["pipeline"].predict_proba(fixtures[model["feature_cols"]])
    classes = list(model["pipeline"].classes_)
    out = {}
    for i, (_, r) in enumerate(fixtures.iterrows()):
        out[(r["home_team"], r["away_team"])] = dict(zip(classes, proba[i]))
    return out


def _sample_outcome(probs, rng):
    return rng.choice(["A", "D", "H"], p=[probs["A"], probs["D"], probs["H"]])


def _knockout_winner(team_a, team_b, elo, rng):
    ea = elo.get(team_a, 1500)
    eb = elo.get(team_b, 1500)
    p_a = 1.0 / (1.0 + 10.0 ** ((eb - ea) / 400.0))
    return team_a if rng.random() < p_a else team_b


def simulate(model, fixtures, groups, match_probs, elo, n_sims=N_SIMS_DEFAULT):
    teams = [t for g in groups for t in g]
    counters = {
        "champion": defaultdict(int),
        "final": defaultdict(int),
        "semi": defaultdict(int),
        "qf": defaultdict(int),
        "advance": defaultdict(int),
    }

    group_match_pairs = {}
    for gi, g in enumerate(groups):
        gset = set(g)
        pairs = [(r["home_team"], r["away_team"]) for _, r in fixtures.iterrows()
                 if r["home_team"] in gset and r["away_team"] in gset]
        group_match_pairs[gi] = pairs

    rng = RNG
    for _ in range(n_sims):
        third_place_pool = []
        qualified = []

        for gi, g in enumerate(groups):
            pts = {t: 0 for t in g}
            for (h, a) in group_match_pairs[gi]:
                outcome = _sample_outcome(match_probs[(h, a)], rng)
                if outcome == "H":
                    pts[h] += 3
                elif outcome == "A":
                    pts[a] += 3
                else:
                    pts[h] += 1
                    pts[a] += 1
            standing = sorted(g, key=lambda t: (-pts[t], -elo.get(t, 1500)))
            qualified.extend(standing[:2])
            third_place_pool.append((standing[2], pts[standing[2]], elo.get(standing[2], 1500)))

        third_place_pool.sort(key=lambda x: (-x[1], -x[2]))
        qualified.extend([t for t, _, _ in third_place_pool[:8]])

        for t in qualified:
            counters["advance"][t] += 1

        bracket = qualified.copy()
        rng.shuffle(bracket)
        next_round = []
        for i in range(0, 32, 2):
            w = _knockout_winner(bracket[i], bracket[i + 1], elo, rng)
            next_round.append(w)
        rng.shuffle(next_round)
        qf_teams = next_round
        for t in qf_teams:
            counters["qf"][t] += 1
        next_round = []
        for i in range(0, 16, 2):
            w = _knockout_winner(qf_teams[i], qf_teams[i + 1], elo, rng)
            next_round.append(w)
        rng.shuffle(next_round)
        sf_teams = next_round
        for t in sf_teams:
            counters["semi"][t] += 1
        next_round = []
        for i in range(0, 8, 2):
            w = _knockout_winner(sf_teams[i], sf_teams[i + 1], elo, rng)
            next_round.append(w)
        finalists = next_round
        for t in finalists:
            counters["final"][t] += 1
        rng.shuffle(finalists)
        champ = _knockout_winner(finalists[0], finalists[1], elo, rng)
        counters["champion"][champ] += 1

    probs = {}
    for k, c in counters.items():
        probs[k] = {t: c[t] / n_sims for t in teams}
    return probs


def main(n_sims=N_SIMS_DEFAULT):
    d = load_data()
    played, fixtures, state = build_features(d["played"], d["fixtures"])
    elo = state["elo"]
    model = joblib.load(ROOT / "models" / "predictor.pkl")

    wc = fixtures[fixtures["tournament"] == "FIFA World Cup"].reset_index(drop=True)
    groups = detect_groups(wc)
    print(f"Detected {len(groups)} groups of {len(groups[0])} teams.")
    for i, g in enumerate(groups):
        print(f"  Group {chr(ord('A')+i)}: {', '.join(g)}")

    match_probs = predict_group_matches(model, wc)

    print(f"\nRunning {n_sims} Monte Carlo simulations...")
    probs = simulate(model, wc, groups, match_probs, elo, n_sims=n_sims)

    print("\nTop 15 by champion probability:")
    top = sorted(probs["champion"].items(), key=lambda kv: -kv[1])[:15]
    for t, p in top:
        print(f"  {t:25s}  champion={p*100:5.2f}%  final={probs['final'][t]*100:5.2f}%  "
              f"SF={probs['semi'][t]*100:5.2f}%  QF={probs['qf'][t]*100:5.2f}%  "
              f"adv={probs['advance'][t]*100:5.2f}%")

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for t in probs["champion"]:
        rows.append({
            "team": t,
            "elo": elo.get(t, 1500),
            "p_advance": probs["advance"][t],
            "p_qf": probs["qf"][t],
            "p_semi": probs["semi"][t],
            "p_final": probs["final"][t],
            "p_champion": probs["champion"][t],
        })
    sim_df = pd.DataFrame(rows).sort_values("p_champion", ascending=False)
    sim_df.to_csv(out_dir / "simulation_results.csv", index=False)

    match_rows = []
    classes = list(model["pipeline"].classes_)
    proba = model["pipeline"].predict_proba(wc[model["feature_cols"]])
    for i, (_, r) in enumerate(wc.iterrows()):
        match_rows.append({
            "date": r["date"].date(),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "city": r["city"],
            "P_home_win": proba[i][classes.index("H")],
            "P_draw":     proba[i][classes.index("D")],
            "P_away_win": proba[i][classes.index("A")],
        })
    matches_df = pd.DataFrame(match_rows)
    matches_df.to_csv(out_dir / "match_predictions.csv", index=False)

    grp_rows = []
    for i, g in enumerate(groups):
        for t in g:
            grp_rows.append({"group": chr(ord("A") + i), "team": t,
                             "elo": elo.get(t, 1500),
                             "p_advance": probs["advance"][t]})
    pd.DataFrame(grp_rows).to_csv(out_dir / "groups.csv", index=False)
    print(f"\nSaved -> {out_dir}/simulation_results.csv, match_predictions.csv, groups.csv")
    return probs, groups, sim_df, matches_df


if __name__ == "__main__":
    main()
