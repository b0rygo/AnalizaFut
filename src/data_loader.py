from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _apply_former_names(df, former):
    df = df.copy()
    for _, row in former.iterrows():
        mask_window = (df["date"] >= row["start_date"]) & (df["date"] <= row["end_date"])
        for col in ("home_team", "away_team", "team", "winner"):
            if col in df.columns:
                df.loc[mask_window & (df[col] == row["former"]), col] = row["current"]
    return df


def load_data(data_dir=DATA_DIR):
    results = pd.read_csv(data_dir / "results.csv", parse_dates=["date"])
    goalscorers = pd.read_csv(data_dir / "goalscorers.csv", parse_dates=["date"])
    shootouts = pd.read_csv(data_dir / "shootouts.csv", parse_dates=["date"])
    former = pd.read_csv(
        data_dir / "former_names.csv", parse_dates=["start_date", "end_date"]
    )

    results = _apply_former_names(results, former)
    goalscorers = _apply_former_names(goalscorers, former)
    shootouts = _apply_former_names(shootouts, former)

    played_mask = results["home_score"].notna() & results["away_score"].notna()
    played = results.loc[played_mask].copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    played["outcome"] = "D"
    played.loc[played["home_score"] > played["away_score"], "outcome"] = "H"
    played.loc[played["home_score"] < played["away_score"], "outcome"] = "A"
    played = played.sort_values("date").reset_index(drop=True)

    fixtures = results.loc[~played_mask].copy().sort_values("date").reset_index(drop=True)

    return {
        "played": played,
        "fixtures": fixtures,
        "goalscorers": goalscorers,
        "shootouts": shootouts,
        "former": former,
    }


if __name__ == "__main__":
    d = load_data()
    print(f"played:    {d['played'].shape}  ({d['played']['date'].min().date()} -> {d['played']['date'].max().date()})")
    print(f"fixtures:  {d['fixtures'].shape}")
    print(f"goalscorers: {d['goalscorers'].shape}")
    print(f"shootouts:   {d['shootouts'].shape}")
    print(d["played"]["outcome"].value_counts(normalize=True).round(3))
