import re
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# =============================================================================
# 1. LOAD DATA
# =============================================================================
frames = []
for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    f = pd.read_csv(f"SEM8/Grad Project/wvb_teammatch_div1_{year}.csv")
    frames.append(f)

df = pd.concat(frames, ignore_index=True)

df = df.rename(columns={
    "Total Attacks": "attempts", "Hit Pct": "hit_pct",
    "Date": "date_raw", "Team": "team", "Opponent": "opponent",
    "Kills": "kills", "Errors": "errors",
})

for c in ["MS", "TB", "TeamID"]:
    if c in df.columns:
        df = df.drop(columns=[c])


# =============================================================================
# 2. CLEAN DATA
# =============================================================================
df = df[df["Result"].str.match(r"^[WL]", na=False)].copy()
df["win"] = df["Result"].str.startswith("W").astype(int)

def clean_opponent(name):
    s = str(name)
    s = re.sub(r"^@\s*", "", s)
    s = re.sub(r"^#\d+\s+", "", s)
    s = re.sub(r"\s+@.*$", "", s)
    s = re.sub(r"\s+\d{4} D1\s*WVB.*$", "", s)
    s = re.sub(r"\s+NCAA Division.*$", "", s)
    return s.strip()

df["opp_clean"] = df["opponent"].apply(clean_opponent)

known_teams = set(df["team"].unique())
df = df[df["opp_clean"].isin(known_teams)].copy()


# =============================================================================
# 3. KEEP PAIRED MATCHES
# =============================================================================
df["match_key"] = (
    df[["team", "opp_clean"]]
    .apply(lambda x: "_".join(sorted(x)), axis=1)
    + "_" + df["date_raw"].astype(str)
)

paired_keys = df["match_key"].value_counts()
paired_keys = paired_keys[paired_keys == 2].index
df = df[df["match_key"].isin(paired_keys)].copy()


# =============================================================================
# 4. DATE + SETS
# =============================================================================
df["date"] = pd.to_datetime(
    df["date_raw"].str.replace(r"\(\d+\)$", "", regex=True).str.strip(),
    format="mixed",
    errors="coerce",
)
df = df.dropna(subset=["date"]).copy()

df = df.sort_values(["team", "date"]).reset_index(drop=True)
df.columns = df.columns.str.lower()

def parse_sets(result):
    try:
        score = str(result).split()[1]
        w, l = score.split("-")
        return int(w), int(l)
    except:
        return np.nan, np.nan

df[["sets_won", "sets_lost"]] = df["result"].apply(lambda r: pd.Series(parse_sets(r)))
df["set_margin"] = df["sets_won"] - df["sets_lost"]
df["sets_played"] = df["sets_won"] + df["sets_lost"]


# =============================================================================
# 5. CORE FEATURES
# =============================================================================
for col in ["kills", "errors", "attempts", "aces", "assists", "digs"]:
    df[f"cum_{col}"] = df.groupby("team")[col].transform(lambda x: x.expanding().sum())

df["cum_hit_pct"] = (df["cum_kills"] - df["cum_errors"]) / df["cum_attempts"].replace(0, np.nan)

df["cum_hit_pct_prev"] = (
    df.groupby("team")["kills"].transform(lambda x: x.shift().expanding().sum()) -
    df.groupby("team")["errors"].transform(lambda x: x.shift().expanding().sum())
) / df.groupby("team")["attempts"].transform(lambda x: x.shift().expanding().sum()).replace(0, np.nan)

for col in ["kills", "errors", "aces", "assists", "digs"]:
    df[f"cum_{col}_prev"] = df.groupby("team")[col].transform(
        lambda x: x.shift().expanding().sum()
    )


# =============================================================================
# 6. ELO
# =============================================================================
K = 32
elo = defaultdict(lambda: 1500)

df = df.sort_values("date").reset_index(drop=True)

team_elo, opp_elo = [], []

for _, row in df.iterrows():
    t, o = row["team"], row["opp_clean"]
    t_elo, o_elo = elo[t], elo[o]

    team_elo.append(t_elo)
    opp_elo.append(o_elo)

    expected = 1 / (1 + 10 ** ((o_elo - t_elo) / 400))
    result = row["sets_won"] / (row["sets_won"] + row["sets_lost"])

    elo[t] += K * (result - expected)
    elo[o] += K * ((1 - result) - (1 - expected))

df["team_elo"] = team_elo
df["opp_elo"] = opp_elo
df["elo_diff"] = df["team_elo"] - df["opp_elo"]


# =============================================================================
# 7. FEATURE SETS (FULL ABLATION)
# =============================================================================
feature_sets = {

    # "box_score_only": [
    #     "cum_hit_pct_prev",
    #     "cum_kills_prev",
    #     "cum_errors_prev",
    #     "cum_aces_prev",
    #     "cum_assists_prev",
    #     "cum_digs_prev"
    # ],

    # "elo_only": [
    #     "team_elo", "opp_elo", "elo_diff"
    # ],

    # "box_plus_elo": [
    #     "cum_hit_pct_prev",
    #     "cum_kills_prev",
    #     "cum_errors_prev",
    #     "cum_aces_prev",
    #     "cum_assists_prev",
    #     "cum_digs_prev",
    #     "team_elo", "opp_elo", "elo_diff"
    # ],

    # "plus_context": [
    #     "cum_hit_pct_prev",
    #     "cum_kills_prev",
    #     "cum_errors_prev",
    #     "cum_aces_prev",
    #     "cum_assists_prev",
    #     "cum_digs_prev",
    #     "team_elo", "opp_elo", "elo_diff",
    #     "sets_played"
    # ],

    "full_model": [
        "cum_hit_pct_prev",
        "cum_kills_prev",
        "cum_errors_prev",
        "cum_aces_prev",
        "cum_assists_prev",
        "cum_digs_prev",
        "team_elo",
        "opp_elo",
        "elo_diff"
]
}


# =============================================================================
# 8. MODELS
# =============================================================================
models = {
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "gbm": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "xgb": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# =============================================================================
# 9. WALK-FORWARD + ABLATION
# =============================================================================
seasons = sorted(df["season"].unique())
ablation_results = {}

for fs_name, fs_features in feature_sets.items():

    print("\n" + "="*60)
    print(f"ABLATION: {fs_name}")
    print("="*60)

    results = {m: {"preds": [], "probs": [], "true": []} for m in models}

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train = df[df["season"].isin(train_seasons)]
        test = df[df["season"] == test_season]

        X_train = train[fs_features].copy()
        y_train = train["win"]
        X_test = test[fs_features].copy()
        y_test = test["win"]

        # === YOUR ORIGINAL NaN LOGIC ===
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]

        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )

        for name, model in models.items():

            model.fit(X_tr, y_tr)

            probs = model.predict_proba(X_test)[:, 1]
            val_probs = model.predict_proba(X_val)[:, 1]

            best_t, best_acc = 0.5, 0
            for t in np.arange(0.3, 0.7, 0.01):
                acc = ((val_probs >= t).astype(int) == y_val.values).mean()
                if acc > best_acc:
                    best_acc, best_t = acc, t

            preds = (probs >= best_t).astype(int)

            results[name]["probs"].extend(probs)
            results[name]["preds"].extend(preds)
            results[name]["true"].extend(y_test)

    ablation_results[fs_name] = {}

    for name in results:
        y_true = results[name]["true"]
        y_pred = results[name]["preds"]
        y_prob = results[name]["probs"]

        acc = (np.array(y_true) == np.array(y_pred)).mean()
        auc = roc_auc_score(y_true, y_prob)

        ablation_results[fs_name][name] = (acc, auc)

        print(f"{name.upper()} → Acc: {acc:.4f}, AUC: {auc:.4f}")


# =============================================================================
# 10. FINAL SUMMARY (XGBOOST)
# =============================================================================
print("\n" + "="*60)
print("FINAL ABLATION (XGBOOST)")
print("="*60)

for fs in ablation_results:
    acc, auc = ablation_results[fs]["xgb"]
    print(f"{fs:20s} | Acc: {acc:.4f} | AUC: {auc:.4f}")