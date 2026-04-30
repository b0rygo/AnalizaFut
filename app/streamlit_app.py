import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_data
from features import FEATURE_COLS, TIER_COL, build_features, build_feature_row

st.set_page_config(page_title="AnalizaFUT", layout="wide")


@st.cache_resource
def _load():
    pkl = ROOT / "models" / "predictor.pkl"
    if not pkl.exists():
        with st.spinner("Trening modelu (jednorazowo, ~30 s)..."):
            from model import main as train_main
            train_main()

    sim_csv = ROOT / "reports" / "simulation_results.csv"
    if not sim_csv.exists():
        with st.spinner("Symulacja Monte Carlo (~10 s)..."):
            from simulate_wc import main as sim_main
            sim_main()

    d = load_data()
    played, fixtures, state = build_features(d["played"], d["fixtures"])
    model = joblib.load(pkl)
    sim = pd.read_csv(sim_csv)
    matches = pd.read_csv(ROOT / "reports" / "match_predictions.csv")
    groups = pd.read_csv(ROOT / "reports" / "groups.csv")
    return played, state, model, sim, matches, groups


played, state, model, sim_df, matches_df, groups_df = _load()
TEAMS = sorted(set(played["home_team"]).union(played["away_team"]))
CLASSES = list(model["pipeline"].classes_)
EXPANDED = model["expanded_names"]
LR_COEF = model["lr_coef"]
LR_INTERCEPT = model["lr_intercept"]


def _short(name):
    return name.replace("num__", "").replace("tier__tier_", "tier=")


def _last_n_summary(team, n=5):
    m = played[(played["home_team"] == team) | (played["away_team"] == team)].tail(n)
    rows = []
    for _, r in m.iterrows():
        if r["home_team"] == team:
            opp, gf, ga = r["away_team"], r["home_score"], r["away_score"]
        else:
            opp, gf, ga = r["home_team"], r["away_score"], r["home_score"]
        res = "W" if gf > ga else ("L" if gf < ga else "D")
        rows.append({"data": r["date"].date(), "rywal": opp,
                     "gole": f"{gf}:{ga}", "wynik": res})
    return pd.DataFrame(rows)


def _h2h_summary(home, away, n=5):
    m = played[
        ((played["home_team"] == home) & (played["away_team"] == away))
        | ((played["home_team"] == away) & (played["away_team"] == home))
    ].tail(n).copy()
    if m.empty:
        return None, (0, 0, 0)
    w_h = ((m["home_team"] == home) & (m["home_score"] > m["away_score"])).sum() \
        + ((m["away_team"] == home) & (m["away_score"] > m["home_score"])).sum()
    w_a = ((m["home_team"] == away) & (m["home_score"] > m["away_score"])).sum() \
        + ((m["away_team"] == away) & (m["away_score"] > m["home_score"])).sum()
    d = (m["home_score"] == m["away_score"]).sum()
    return m, (int(w_h), int(d), int(w_a))


st.title("AnalizaFUT")
tab1, tab2, tab3 = st.tabs(["Predyktor", "World Cup 2026", "Model"])


with tab1:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Gospodarz", TEAMS, index=TEAMS.index("Brazil"))
    away = c2.selectbox("Goście", TEAMS,
                        index=TEAMS.index("Argentina") if "Argentina" in TEAMS else 0)
    c3, c4 = st.columns(2)
    tier = c3.selectbox("Ranga", ["World Cup", "Continental", "Qualifier", "Friendly", "Other"])
    neutral = c4.checkbox("Boisko neutralne", value=True)

    if home == away:
        st.stop()

    X = build_feature_row(state, home, away, tier=tier,
                          neutral=neutral, host_country=home if not neutral else None)
    X_in = X[FEATURE_COLS + [TIER_COL]]

    proba_gbm = model["pipeline"].predict_proba(X_in)[0]
    proba_lr = model["lr_pipeline"].predict_proba(X_in)[0]
    pH = proba_gbm[CLASSES.index("H")]
    pD = proba_gbm[CLASSES.index("D")]
    pA = proba_gbm[CLASSES.index("A")]

    st.markdown(f"**{home} {pH:.1%}** · remis {pD:.1%} · **{away} {pA:.1%}**")
    edge = abs(pH - pA)
    if pH > pA + 0.05:
        st.write(f"Faworyt: {home} (różnica {edge:.0%}).")
    elif pA > pH + 0.05:
        st.write(f"Faworyt: {away} (różnica {edge:.0%}).")
    else:
        st.write("Mecz wyrównany.")

    st.divider()
    st.markdown("### Pod maską modelu (regresja logistyczna)")
    st.caption("η_k = b_k + Σ w_k,i · z_i, softmax(η) → P.")

    pre = model["lr_pipeline"].named_steps["pre"]
    z = pre.transform(X_in)[0]
    z_dict = dict(zip(EXPANDED, z))

    contrib = []
    for feat, zi in z_dict.items():
        row = {"cecha": _short(feat), "z (std.)": zi}
        for cls in CLASSES:
            w = LR_COEF[cls][feat]
            row[f"w·z ({cls})"] = w * zi
        contrib.append(row)
    contrib_df = pd.DataFrame(contrib)
    contrib_df["|w·z| max"] = contrib_df[[f"w·z ({c})" for c in CLASSES]].abs().max(axis=1)
    contrib_df = contrib_df.sort_values("|w·z| max", ascending=False).drop(columns="|w·z| max")

    logit = {cls: LR_INTERCEPT[cls] + sum(LR_COEF[cls][f] * z_dict[f] for f in EXPANDED)
             for cls in CLASSES}

    summary = pd.DataFrame({
        "klasa": CLASSES,
        "intercept b_k": [LR_INTERCEPT[c] for c in CLASSES],
        "Σ w·z": [logit[c] - LR_INTERCEPT[c] for c in CLASSES],
        "logit η_k": [logit[c] for c in CLASSES],
        "P (LR)": [proba_lr[CLASSES.index(c)] for c in CLASSES],
        "P (GBM, prod.)": [proba_gbm[CLASSES.index(c)] for c in CLASSES],
    })
    st.dataframe(
        summary.style.format({
            "intercept b_k": "{:+.3f}", "Σ w·z": "{:+.3f}", "logit η_k": "{:+.3f}",
            "P (LR)": "{:.1%}", "P (GBM, prod.)": "{:.1%}",
        }),
        hide_index=True, use_container_width=True,
    )

    st.markdown("**Wkład cech do logitów (top 10 wg |w·z|)**")
    st.dataframe(
        contrib_df.head(10).style.format({
            "z (std.)": "{:+.3f}",
            "w·z (A)": "{:+.3f}", "w·z (D)": "{:+.3f}", "w·z (H)": "{:+.3f}",
        }),
        hide_index=True, use_container_width=True,
    )

    with st.expander("Pełna tabela wkładów (20 cech × 3 klasy)"):
        st.dataframe(
            contrib_df.style.format({
                "z (std.)": "{:+.3f}",
                "w·z (A)": "{:+.3f}", "w·z (D)": "{:+.3f}", "w·z (H)": "{:+.3f}",
            }),
            hide_index=True, use_container_width=True,
        )

    with st.expander("Surowe wartości cech wejściowych"):
        st.dataframe(
            X_in.T.rename(columns={X_in.index[0]: "wartość"}),
            use_container_width=True,
        )

    st.divider()
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"**{home} — 5 ostatnich**")
        f_h = _last_n_summary(home, 5)
        if not f_h.empty:
            st.dataframe(f_h, hide_index=True, use_container_width=True)
    with fc2:
        st.markdown(f"**{away} — 5 ostatnich**")
        f_a = _last_n_summary(away, 5)
        if not f_a.empty:
            st.dataframe(f_a, hide_index=True, use_container_width=True)

    h2h_df, (wH, dD, wA) = _h2h_summary(home, away, 5)
    st.markdown(f"**Bezpośrednio (5 ost.):** {home} {wH} – {dD} – {wA} {away}")
    if h2h_df is not None:
        view = h2h_df.copy()
        view["data"] = view["date"].dt.date
        st.dataframe(
            view[["data", "home_team", "home_score", "away_score", "away_team", "tournament"]],
            hide_index=True, use_container_width=True,
        )


with tab2:
    sim = sim_df.sort_values("p_champion", ascending=False).reset_index(drop=True)
    elo_map = state["elo"]
    sim["elo"] = sim["team"].map(elo_map).fillna(sim["elo"])
    z = np.exp(sim["elo"].to_numpy() / 200.0)
    sim["elo_implied"] = z / z.sum()
    sim["delta"] = sim["p_champion"] - sim["elo_implied"]

    st.subheader("Faworyci do tytułu")
    show = sim.head(15)[["team", "elo", "p_advance", "p_qf", "p_semi", "p_final", "p_champion", "elo_implied", "delta"]]
    st.dataframe(
        show.rename(columns={
            "team": "drużyna", "elo": "Elo",
            "p_advance": "awans", "p_qf": "ćwierć", "p_semi": "półfinał",
            "p_final": "finał", "p_champion": "mistrz",
            "elo_implied": "Elo-implied", "delta": "Δ vs Elo",
        }).style.format({
            "Elo": "{:.0f}",
            "awans": "{:.0%}", "ćwierć": "{:.0%}", "półfinał": "{:.0%}",
            "finał": "{:.0%}", "mistrz": "{:.1%}",
            "Elo-implied": "{:.1%}", "Δ vs Elo": "{:+.1%}",
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption("Δ vs Elo = przewaga/strata Monte Carlo nad softmax(Elo/200).")

    st.divider()
    st.subheader("Najciekawsze mecze fazy grupowej (entropia)")
    m = matches_df.copy()
    p = m[["P_home_win", "P_draw", "P_away_win"]].to_numpy().clip(1e-9, 1)
    m["H (nat)"] = -(p * np.log(p)).sum(axis=1)
    m["mecz"] = m["home_team"] + " – " + m["away_team"]
    top_close = m.nlargest(8, "H (nat)")[["date", "mecz", "P_home_win", "P_draw", "P_away_win", "H (nat)"]]
    st.dataframe(
        top_close.rename(columns={"date": "data", "P_home_win": "1", "P_draw": "X", "P_away_win": "2"})
            .style.format({"1": "{:.0%}", "X": "{:.0%}", "2": "{:.0%}", "H (nat)": "{:.3f}"}),
        hide_index=True, use_container_width=True,
    )
    st.caption("H = -Σ p·ln p. Maksimum dla rozkładu (1/3,1/3,1/3) ≈ 1.099.")

    st.subheader("Wszystkie mecze")
    pick = st.multiselect("Filtr drużyn", TEAMS, default=[])
    view = matches_df.copy()
    if pick:
        view = view[view["home_team"].isin(pick) | view["away_team"].isin(pick)]
    st.dataframe(
        view.style.format({"P_home_win": "{:.0%}", "P_draw": "{:.0%}", "P_away_win": "{:.0%}"}),
        hide_index=True, use_container_width=True, height=380,
    )

    st.divider()
    st.subheader("Grupy")
    cols = st.columns(4)
    for i, (g, sub) in enumerate(groups_df.groupby("group")):
        with cols[i % 4]:
            st.write(f"**{g}**")
            sub = sub.sort_values("p_advance", ascending=False)
            st.dataframe(
                sub[["team", "p_advance"]].rename(columns={"team": "drużyna", "p_advance": "awans"})
                    .style.format({"awans": "{:.0%}"}),
                hide_index=True, use_container_width=True,
            )


with tab3:
    st.subheader("Porównanie modeli (hold-out 2023-01-01 → 2026-03-31)")
    all_metrics = model["test_metrics"]["all_models"]
    metrics_df = pd.DataFrame([
        {"model": n, "accuracy": v["accuracy"], "log_loss": v["log_loss"]}
        for n, v in all_metrics.items()
    ])
    metrics_df["Δ acc vs baseline"] = metrics_df["accuracy"] - 0.4685
    st.dataframe(
        metrics_df.style.format({
            "accuracy": "{:.4f}", "log_loss": "{:.4f}", "Δ acc vs baseline": "{:+.4f}",
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption(f"Wybrany do produkcji: **{model['best_model']}** (najniższy log-loss). "
               "Baseline = always-H, accuracy 0.4685.")

    st.divider()
    st.subheader("Confusion matrix — best model na hold-out")
    cm = np.array(model["test_metrics"]["confusion_matrix"])
    labels = model["test_metrics"]["cm_labels"]
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels])
    cm_df["row_total"] = cm_df.sum(axis=1)
    cm_df.loc["col_total"] = cm_df.sum(axis=0)
    st.dataframe(cm_df, use_container_width=True)
    diag = np.trace(cm)
    total = cm.sum()
    st.caption(f"Trafnych: {diag}/{total} = {diag/total:.4f}.")

    st.divider()
    c_l, c_r = st.columns(2)
    with c_l:
        st.subheader("GBM — feature importances")
        imp = pd.DataFrame(model["gbm_importances"], columns=["cecha", "ważność"])
        imp["cecha"] = imp["cecha"].map(_short)
        imp = imp.sort_values("ważność", ascending=False)
        st.bar_chart(imp.set_index("cecha"), height=420)
        st.caption("Mean decrease in impurity.")

    with c_r:
        st.subheader("LR — współczynniki w_k,i")
        coef_rows = []
        for feat in EXPANDED:
            r = {"cecha": _short(feat)}
            for cls in CLASSES:
                r[f"w_{cls}"] = LR_COEF[cls][feat]
            coef_rows.append(r)
        coef_df = pd.DataFrame(coef_rows)
        coef_df["|w| max"] = coef_df[[f"w_{c}" for c in CLASSES]].abs().max(axis=1)
        coef_df = coef_df.sort_values("|w| max", ascending=False).drop(columns="|w| max")
        st.dataframe(
            coef_df.style.format({"w_A": "{:+.3f}", "w_D": "{:+.3f}", "w_H": "{:+.3f}"}),
            hide_index=True, use_container_width=True, height=420,
        )
        st.caption(
            f"Intercepty: A={LR_INTERCEPT['A']:+.3f}, D={LR_INTERCEPT['D']:+.3f}, "
            f"H={LR_INTERCEPT['H']:+.3f}."
        )

    st.divider()
    st.subheader("Hiperparametry")
    hp = model["hyperparameters"]
    for name, params in hp.items():
        with st.expander(name):
            small = {k: v for k, v in params.items()
                     if not k.startswith("ccp") and v not in (None,) and not callable(v)}
            st.json(small, expanded=False)
