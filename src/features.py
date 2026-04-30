from collections import defaultdict, deque

import numpy as np
import pandas as pd

DEFAULT_ELO = 1500.0
FORM_WINDOW = 10
H2H_WINDOW = 5

K_BY_TIER = {
    "World Cup": 60.0,
    "Continental": 50.0,
    "Qualifier": 40.0,
    "Friendly": 20.0,
    "Other": 30.0,
}

CONTINENTAL_KEYWORDS = (
    "UEFA Euro",
    "Copa América",
    "Copa America",
    "African Cup of Nations",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF",
    "Oceania Nations Cup",
    "Confederations Cup",
)


def tournament_tier(name):
    if not isinstance(name, str):
        return "Other"
    n = name.lower()
    if "fifa world cup" in n and "qualif" not in n:
        return "World Cup"
    if "qualif" in n:
        return "Qualifier"
    if n == "friendly":
        return "Friendly"
    for kw in CONTINENTAL_KEYWORDS:
        if kw.lower() in n:
            return "Continental"
    return "Other"


def _elo_expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _goal_diff_multiplier(goal_diff):
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def build_features(played, fixtures):
    played = played.copy()
    fixtures = fixtures.copy()
    played["__src"] = "played"
    fixtures["__src"] = "fixtures"
    all_matches = pd.concat([played, fixtures], ignore_index=True)
    all_matches = all_matches.sort_values(["date"], kind="mergesort").reset_index(drop=True)

    elo = defaultdict(lambda: DEFAULT_ELO)
    last_n_results = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    h2h = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    last_date = {}

    feats = []
    for _, m in all_matches.iterrows():
        h, a = m["home_team"], m["away_team"]
        date = m["date"]
        tier = tournament_tier(m["tournament"])
        neutral = bool(m["neutral"])
        home_adv = (not neutral) and (m["country"] == h)

        elo_h, elo_a = elo[h], elo[a]

        def _form(team):
            buf = last_n_results[team]
            if not buf:
                return 0.0, 0.0, 0.0
            arr = np.asarray(buf, dtype=float)
            return arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()

        form_pts_h, form_gf_h, form_ga_h = _form(h)
        form_pts_a, form_gf_a, form_ga_a = _form(a)

        pair = tuple(sorted([h, a]))
        h2h_buf = h2h[pair]
        if h2h_buf:
            h2h_home_pts = 0.0
            h2h_away_pts = 0.0
            for row in h2h_buf:
                prev_home, hp, ap = row
                if prev_home == h:
                    h2h_home_pts += hp
                    h2h_away_pts += ap
                else:
                    h2h_home_pts += ap
                    h2h_away_pts += hp
            h2h_home_pts /= len(h2h_buf)
            h2h_away_pts /= len(h2h_buf)
        else:
            h2h_home_pts = h2h_away_pts = 0.0

        days_since_h = (date - last_date[h]).days if h in last_date else 365
        days_since_a = (date - last_date[a]).days if a in last_date else 365
        days_since_h = min(days_since_h, 365)
        days_since_a = min(days_since_a, 365)

        feats.append({
            "elo_home": elo_h,
            "elo_away": elo_a,
            "elo_diff": elo_h - elo_a,
            "form_pts_home": form_pts_h,
            "form_pts_away": form_pts_a,
            "form_gf_home": form_gf_h,
            "form_gf_away": form_gf_a,
            "form_ga_home": form_ga_h,
            "form_ga_away": form_ga_a,
            "h2h_home_pts": h2h_home_pts,
            "h2h_away_pts": h2h_away_pts,
            "tier": tier,
            "neutral": int(neutral),
            "home_advantage": int(home_adv),
            "days_since_home": days_since_h,
            "days_since_away": days_since_a,
        })

        if m["__src"] == "played":
            hs, as_ = int(m["home_score"]), int(m["away_score"])
            if hs > as_:
                pts_h, pts_a, score_h = 3, 0, 1.0
            elif hs < as_:
                pts_h, pts_a, score_h = 0, 3, 0.0
            else:
                pts_h, pts_a, score_h = 1, 1, 0.5

            k = K_BY_TIER.get(tier, 30.0)
            mult = _goal_diff_multiplier(hs - as_)
            exp_h = _elo_expected(elo_h, elo_a)
            delta = k * mult * (score_h - exp_h)
            elo[h] = elo_h + delta
            elo[a] = elo_a - delta

            last_n_results[h].append((pts_h, hs, as_))
            last_n_results[a].append((pts_a, as_, hs))
            h2h[pair].append((h, pts_h, pts_a))
            last_date[h] = date
            last_date[a] = date

    feat_df = pd.DataFrame(feats)
    base = all_matches.reset_index(drop=True).drop(columns=["neutral"], errors="ignore")
    enriched = pd.concat([base, feat_df], axis=1)

    played_out = enriched[enriched["__src"] == "played"].drop(columns="__src").reset_index(drop=True)
    fixtures_out = enriched[enriched["__src"] == "fixtures"].drop(columns="__src").reset_index(drop=True)

    state = {
        "elo": dict(elo),
        "form": {t: list(buf) for t, buf in last_n_results.items()},
        "h2h": {pair: list(buf) for pair, buf in h2h.items()},
        "last_date": dict(last_date),
        "as_of": all_matches["date"].max(),
    }
    return played_out, fixtures_out, state


def build_feature_row(state, home, away, *, tier="Friendly", neutral=True,
                      host_country=None, match_date=None):
    elo = state["elo"]
    form = state["form"]
    h2h = state["h2h"]
    last_date = state["last_date"]
    if match_date is None:
        match_date = state["as_of"]

    elo_h = elo.get(home, DEFAULT_ELO)
    elo_a = elo.get(away, DEFAULT_ELO)

    def _form(team):
        buf = form.get(team, [])
        if not buf:
            return 0.0, 0.0, 0.0
        arr = np.asarray(buf, dtype=float)
        return arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()

    fp_h, gf_h, ga_h = _form(home)
    fp_a, gf_a, ga_a = _form(away)

    pair = tuple(sorted([home, away]))
    buf = h2h.get(pair, [])
    if buf:
        hp = ap = 0.0
        for prev_home, php, pap in buf:
            if prev_home == home:
                hp += php; ap += pap
            else:
                hp += pap; ap += php
        h2h_h, h2h_a = hp / len(buf), ap / len(buf)
    else:
        h2h_h = h2h_a = 0.0

    ds_h = min((pd.Timestamp(match_date) - last_date[home]).days, 365) if home in last_date else 365
    ds_a = min((pd.Timestamp(match_date) - last_date[away]).days, 365) if away in last_date else 365

    home_adv = (not neutral) and (host_country is not None) and (host_country == home)

    return pd.DataFrame([{
        "elo_home": elo_h, "elo_away": elo_a, "elo_diff": elo_h - elo_a,
        "form_pts_home": fp_h, "form_pts_away": fp_a,
        "form_gf_home": gf_h, "form_gf_away": gf_a,
        "form_ga_home": ga_h, "form_ga_away": ga_a,
        "h2h_home_pts": h2h_h, "h2h_away_pts": h2h_a,
        "tier": tier,
        "neutral": int(neutral), "home_advantage": int(home_adv),
        "days_since_home": ds_h, "days_since_away": ds_a,
    }])


FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "form_pts_home", "form_pts_away",
    "form_gf_home", "form_gf_away",
    "form_ga_home", "form_ga_away",
    "h2h_home_pts", "h2h_away_pts",
    "neutral", "home_advantage",
    "days_since_home", "days_since_away",
]
TIER_COL = "tier"


if __name__ == "__main__":
    from data_loader import load_data
    d = load_data()
    p, f, state = build_features(d["played"], d["fixtures"])
    print(f"played: {p.shape}, fixtures: {f.shape}")
    top = sorted(state["elo"].items(), key=lambda kv: -kv[1])[:15]
    for t, e in top:
        print(f"  {t:30s} {e:7.1f}")
