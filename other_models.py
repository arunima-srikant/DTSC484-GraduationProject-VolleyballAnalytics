import re
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =============================================================================
# 1. LOAD & COMBINE RAW SOURCE FILES
# =============================================================================
# Unzip — adjust path as needed
#zipfile.ZipFile("wvb_teammatch_div1_2024.zip").extractall(".")

frames = []
for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    f = pd.read_csv(f"SEM8/Grad Project/wvb_teammatch_div1_{year}.csv")
    frames.append(f)

df = pd.concat(frames, ignore_index=True)

# Standardize column names
df = df.rename(columns={
    "Total Attacks": "attempts", "Hit Pct": "hit_pct",
    "Date": "date_raw", "Team": "team", "Opponent": "opponent",
    "Kills": "kills", "Errors": "errors",
})
for c in ["MS", "TB", "TeamID"]:
    if c in df.columns:
        df = df.drop(columns=[c])

# =============================================================================
# 2. FILTER TO PLAYED MATCHES & CLEAN OPPONENT NAMES
# =============================================================================
df = df[df["Result"].str.match(r"^[WL]", na=False)].copy()
df["win"] = df["Result"].str.startswith("W").astype(int)


def clean_opponent(name):
    """Strip away-game prefixes, rankings, and tournament/location suffixes."""
    s = str(name)
    s = re.sub(r"^@\s*", "", s)           # '@ Nebraska' -> 'Nebraska'
    s = re.sub(r"^#\d+\s+", "", s)        # '#1 Nebraska' -> 'Nebraska'
    s = re.sub(r"\s+@.*$", "", s)          # 'Nebraska @Lincoln, NE (...)' -> 'Nebraska'
    s = re.sub(r"\s+\d{4} D1\s*WVB.*$", "", s)  # 'Nebraska 2025 D1WVB ...' -> 'Nebraska'
    s = re.sub(r"\s+NCAA Division.*$", "", s)     # 'Nebraska NCAA Division ...' -> 'Nebraska'
    return s.strip()


df["opp_clean"] = df["opponent"].apply(clean_opponent)

# Keep only rows where opponent is a tracked team
known_teams = set(df["team"].unique())
before = len(df)
df = df[df["opp_clean"].isin(known_teams)].copy()
print(f"Dropped {before - len(df)} unresolvable opponent rows. Remaining: {len(df)}")


# =============================================================================
# 3. KEEP ONLY PAIRED MATCHES (both team perspectives exist)
# =============================================================================
df["match_key"] = (
    df[["team", "opp_clean"]]
    .apply(lambda x: "_".join(sorted(x)), axis=1)
    + "_" + df["date_raw"].astype(str)
)

key_counts = df["match_key"].value_counts()
paired_keys = key_counts[key_counts == 2].index
before = len(df)
df = df[df["match_key"].isin(paired_keys)].copy()
print(f"Dropped {before - len(df)} unpaired rows. Remaining: {len(df)}")


# =============================================================================
# 4. PARSE DATE & SETS
# =============================================================================
df["date"] = pd.to_datetime(
    df["date_raw"].str.replace(r"\(\d+\)$", "", regex=True).str.strip(),
    format="mixed",
)
df = df.sort_values(["team", "date"]).reset_index(drop=True)

# Lowercase all columns for consistency
df.columns = df.columns.str.lower()


def parse_sets(result):
    try:
        score = str(result).strip().split()[1]
        w, l = score.split("-")
        return int(w), int(l)
    except Exception:
        return np.nan, np.nan


df[["sets_won", "sets_lost"]] = df["result"].apply(
    lambda r: pd.Series(parse_sets(r))
)
df["set_margin"] = df["sets_won"] - df["sets_lost"]
df["sets_played"] = df["sets_won"] + df["sets_lost"]


# =============================================================================
# 5. BUILD CUMULATIVE FEATURES
# =============================================================================
def lookup(d, match_key, team):
    """Safe dict lookup for opponent stats via match_key."""
    return float(d.get((match_key, team), np.nan))


# --- Cumulative raw stats ---
for col, cum_name in [
    ("kills", "cum_kills"), ("errors", "cum_errors"),
    ("attempts", "cum_attempts"), ("aces", "cum_aces"),
    ("assists", "cum_assists"), ("digs", "cum_digs"),
]:
    df[cum_name] = df.groupby("team")[col].transform(
        lambda x: x.expanding().sum()
    )

df["cum_hit_pct"] = (df["cum_kills"] - df["cum_errors"]) / df[
    "cum_attempts"
].replace(0, np.nan)

# --- "prev" versions (shifted = before current match) ---
for col, prev_name in [
    ("cum_kills", "cum_kills_prev"),
    ("cum_errors", "cum_errors_prev"),
    ("cum_aces", "cum_aces_prev"),
    ("cum_assists", "cum_assists_prev"),
    ("cum_digs", "cum_digs_prev"),
]:
    df[prev_name] = df.groupby("team")[col].shift(1)

cum_k = df.groupby("team")["kills"].transform(
    lambda x: x.shift().expanding().sum()
)
cum_e = df.groupby("team")["errors"].transform(
    lambda x: x.shift().expanding().sum()
)
cum_a = df.groupby("team")["attempts"].transform(
    lambda x: x.shift().expanding().sum()
)
df["cum_hit_pct_prev"] = (cum_k - cum_e) / cum_a.replace(0, np.nan)

# --- Opponent cumulative stats via match_key lookup ---
opp_lookups = {
    "opp_cum_hit_pct_prev": "cum_hit_pct_prev",
    "opp_cum_kills_prev": "cum_kills_prev",
    "opp_cum_errors_prev": "cum_errors_prev",
    "opp_cum_aces_prev": "cum_aces_prev",
    "opp_cum_assists_prev": "cum_assists_prev",
    "opp_cum_digs_prev": "cum_digs_prev",
}
for opp_col, src_col in opp_lookups.items():
    d = df.set_index(["match_key", "team"])[src_col].to_dict()
    df[opp_col] = [
        lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
    ]

# --- Diff features ---
df["hit_pct_diff"] = df["cum_hit_pct_prev"] - df["opp_cum_hit_pct_prev"]
df["kills_diff"] = df["cum_kills_prev"] - df["opp_cum_kills_prev"]
df["errors_diff"] = df["cum_errors_prev"] - df["opp_cum_errors_prev"]
df["aces_diff"] = df["cum_aces_prev"] - df["opp_cum_aces_prev"]
df["assists_diff"] = df["cum_assists_prev"] - df["opp_cum_assists_prev"]
df["digs_diff"] = df["cum_digs_prev"] - df["opp_cum_digs_prev"]


# =============================================================================
# 6. ADVANCED FEATURES
# =============================================================================

# --- Home / Away / Neutral ---
def get_location(opponent):
    opponent = str(opponent)
    if opponent.startswith("@"):
        return "Away"
    elif "@" in opponent:
        return "Neutral"
    else:
        return "Home"


df["homeaway"] = df["opponent"].apply(get_location)

# --- Set dominance ---
df["cum_set_win_rate"] = df.groupby("team")["sets_won"].transform(
    lambda x: x.shift().expanding().sum()
) / df.groupby("team")["sets_played"].transform(
    lambda x: x.shift().expanding().sum()
)
df["cum_avg_set_margin"] = df.groupby("team")["set_margin"].transform(
    lambda x: x.shift().expanding().mean()
)
for col in ["cum_set_win_rate", "cum_avg_set_margin"]:
    d = df.set_index(["match_key", "team"])[col].to_dict()
    df[f"opp_{col}"] = [
        lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
    ]
df["set_win_rate_diff"] = df["cum_set_win_rate"] - df["opp_cum_set_win_rate"]
df["set_margin_diff"] = (
    df["cum_avg_set_margin"] - df["opp_cum_avg_set_margin"]
)

# --- Home/Away win rates ---
df["team_home_winrate"] = (
    df.groupby("team")
    .apply(
        lambda g: g["win"]
        .where(g["homeaway"] == "Home")
        .shift()
        .expanding()
        .mean(),
        include_groups=False,
    )
    .reset_index(level=0, drop=True)
)
df["team_away_winrate"] = (
    df.groupby("team")
    .apply(
        lambda g: g["win"]
        .where(g["homeaway"] == "Away")
        .shift()
        .expanding()
        .mean(),
        include_groups=False,
    )
    .reset_index(level=0, drop=True)
)
for col in ["team_home_winrate", "team_away_winrate"]:
    d = df.set_index(["match_key", "team"])[col].to_dict()
    opp_col = "opp_" + col.replace("team_", "")
    df[opp_col] = [
        lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
    ]

# --- Rolling form (multiple windows) ---
for w in [3, 5, 10]:
    col = f"last{w}_win_rate"
    df[col] = df.groupby("team")["win"].transform(
        lambda x: x.shift().rolling(w, min_periods=1).mean()
    )
    d = df.set_index(["match_key", "team"])[col].to_dict()
    df[f"opp_{col}"] = [
        lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
    ]
    df[f"form{w}_diff"] = df[col] - df[f"opp_{col}"]

# --- EWA momentum ---
df["ewa_win_rate"] = df.groupby("team")["win"].transform(
    lambda x: x.shift().ewm(span=5, adjust=False).mean()
)
d = df.set_index(["match_key", "team"])["ewa_win_rate"].to_dict()
df["opp_ewa_win_rate"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["ewa_form_diff"] = df["ewa_win_rate"] - df["opp_ewa_win_rate"]

# --- Rest days ---
df = df.sort_values(["team", "date"])
df["days_since_last"] = df.groupby("team")["date"].diff().dt.days
d = df.set_index(["match_key", "team"])["days_since_last"].to_dict()
df["opp_days_since_last"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["rest_diff"] = df["days_since_last"] - df["opp_days_since_last"]

# --- Elo (season-decayed, margin-scaled, date-snapshot) ---
K = 32
SEASON_DECAY = 0.75
elo = defaultdict(lambda: 1500)
team_elo_list = [0.0] * len(df)
opp_elo_list = [0.0] * len(df)

df = df.sort_values("date").reset_index(drop=True)

prev_season = None
for date, group in df.groupby("date"):
    cur_season = group["season"].iloc[0]
    if prev_season is not None and cur_season != prev_season:
        for t in list(elo.keys()):
            elo[t] = 1500 + SEASON_DECAY * (elo[t] - 1500)
    prev_season = cur_season

    elo_snapshot = dict(elo)
    for idx, row in group.iterrows():
        team = row["team"]
        opp = row["opp_clean"]
        t_rating = elo_snapshot.get(team, 1500)
        o_rating = elo_snapshot.get(opp, 1500)
        team_elo_list[idx] = t_rating
        opp_elo_list[idx] = o_rating

        expected = 1 / (1 + 10 ** ((o_rating - t_rating) / 400))
        result = row["sets_won"] / (row["sets_won"] + row["sets_lost"])
        margin_factor = 1 + abs(row["set_margin"]) / (
            row["sets_played"] + 1e-6
        )
        elo[team] += K * margin_factor * (result - expected)
        elo[opp] += K * margin_factor * ((1 - result) - (1 - expected))

df["team_elo"] = team_elo_list
df["opp_elo"] = opp_elo_list
df["elo_diff"] = df["team_elo"] - df["opp_elo"]

# --- Strength of schedule ---
df["sos"] = df.groupby("team")["opp_elo"].transform(
    lambda x: x.shift().expanding().mean()
)
d = df.set_index(["match_key", "team"])["sos"].to_dict()
df["opp_sos"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["sos_diff"] = df["sos"] - df["opp_sos"]

# --- Opponent-adjusted hitting ---
df["opp_def_pressure"] = df["opp_cum_errors_prev"] / (
    df["opp_cum_kills_prev"] + df["opp_cum_errors_prev"] + 1e-6
)
df["adj_hit_pct"] = df["cum_hit_pct_prev"] - df["opp_def_pressure"]
df["adj_hit_pct_diff"] = df["adj_hit_pct"] - df["opp_cum_hit_pct_prev"]

# --- Head-to-head ---
df = df.sort_values(["team", "date"]).reset_index(drop=True)
df["h2h_wins"] = df.groupby(["team", "opp_clean"])["win"].transform(
    lambda x: x.shift().expanding().sum()
)
df["h2h_games"] = df.groupby(["team", "opp_clean"])["win"].transform(
    lambda x: x.shift().expanding().count()
)
df["h2h_win_rate"] = df["h2h_wins"] / (df["h2h_games"] + 1e-6)
d = df.set_index(["match_key", "team"])["h2h_win_rate"].to_dict()
df["opp_h2h_win_rate"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["h2h_win_rate_diff"] = df["h2h_win_rate"] - df["opp_h2h_win_rate"]

# --- Season win rate ---
df = df.sort_values(["team", "season", "date"]).reset_index(drop=True)
df["season_wins_so_far"] = df.groupby(["team", "season"])["win"].transform(
    lambda x: x.shift().expanding().sum()
)
df["season_games_so_far"] = df.groupby(["team", "season"])["win"].transform(
    lambda x: x.shift().expanding().count()
)
df["season_win_rate"] = df["season_wins_so_far"] / (
    df["season_games_so_far"] + 1e-6
)
d = df.set_index(["match_key", "team"])["season_win_rate"].to_dict()
df["opp_season_win_rate"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["season_win_rate_diff"] = (
    df["season_win_rate"] - df["opp_season_win_rate"]
)

# --- Consistency (rolling std of set margin) ---
df["set_margin_std"] = df.groupby("team")["set_margin"].transform(
    lambda x: x.shift().rolling(10, min_periods=3).std()
)
d = df.set_index(["match_key", "team"])["set_margin_std"].to_dict()
df["opp_set_margin_std"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["consistency_diff"] = df["set_margin_std"] - df["opp_set_margin_std"]

# --- Cumulative win rate ---
df["cum_win_rate"] = df.groupby("team")["win"].transform(
    lambda x: x.shift().expanding().mean()
)
d = df.set_index(["match_key", "team"])["cum_win_rate"].to_dict()
df["opp_cum_win_rate"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["cum_win_rate_diff"] = df["cum_win_rate"] - df["opp_cum_win_rate"]

# --- Per-set averages ---
df = df.sort_values(["team", "date"]).reset_index(drop=True)
for stat in ["kills", "errors", "aces", "assists", "digs"]:
    cum_stat = df.groupby("team")[stat].transform(
        lambda x: x.shift().expanding().sum()
    )
    cum_sets = df.groupby("team")["s"].transform(
        lambda x: x.shift().expanding().sum()
    )
    df[f"{stat}_per_set"] = cum_stat / cum_sets.replace(0, np.nan)
    d = df.set_index(["match_key", "team"])[f"{stat}_per_set"].to_dict()
    df[f"opp_{stat}_per_set"] = [
        lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
    ]
    df[f"{stat}_per_set_diff"] = (
        df[f"{stat}_per_set"] - df[f"opp_{stat}_per_set"]
    )

# --- Rolling hit pct (5-match window) ---
df["rolling_hit_pct"] = (
    df.groupby("team")["kills"].transform(
        lambda x: x.shift().rolling(5, min_periods=2).sum()
    )
    - df.groupby("team")["errors"].transform(
        lambda x: x.shift().rolling(5, min_periods=2).sum()
    )
) / df.groupby("team")["attempts"].transform(
    lambda x: x.shift().rolling(5, min_periods=2).sum()
).replace(
    0, np.nan
)
d = df.set_index(["match_key", "team"])["rolling_hit_pct"].to_dict()
df["opp_rolling_hit_pct"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["rolling_hit_pct_diff"] = (
    df["rolling_hit_pct"] - df["opp_rolling_hit_pct"]
)

# --- Streak ---
def calc_streak(wins):
    streak, current = [], 0
    for w in wins:
        streak.append(current)
        if w == 1:
            current = current + 1 if current >= 0 else 1
        else:
            current = current - 1 if current <= 0 else -1
    return streak


df = df.sort_values(["team", "date"]).reset_index(drop=True)
df["streak"] = df.groupby("team")["win"].transform(calc_streak)
d = df.set_index(["match_key", "team"])["streak"].to_dict()
df["opp_streak"] = [
    lookup(d, r.match_key, r.opp_clean) for r in df.itertuples()
]
df["streak_diff"] = df["streak"] - df["opp_streak"]

# --- Conference Elo ---
df["conf_avg_elo"] = df.groupby(["date", "conference"])["team_elo"].transform(
    "mean"
)
df["elo_vs_conf"] = df["team_elo"] - df["conf_avg_elo"]

# --- Game number in season ---
df["game_num"] = df.groupby(["team", "season"]).cumcount() + 1


# =============================================================================
# 7. LOCATION DUMMIES
# =============================================================================
loc = pd.get_dummies(df["homeaway"], prefix="loc", drop_first=True, dtype=int)
df = pd.concat([df, loc], axis=1)
loc_cols = [c for c in df.columns if c.startswith("loc_")]


# =============================================================================
# 8. FEATURE LIST & CLEANING
# =============================================================================
core_features = [
    "cum_hit_pct_prev", "cum_kills_prev", "cum_errors_prev",
    "cum_aces_prev", "cum_assists_prev", "cum_digs_prev",
    "opp_cum_hit_pct_prev", "opp_cum_kills_prev", "opp_cum_errors_prev",
    "opp_cum_aces_prev", "opp_cum_assists_prev", "opp_cum_digs_prev",
    "hit_pct_diff", "kills_diff", "errors_diff",
    "aces_diff", "assists_diff", "digs_diff",
    "team_elo", "opp_elo", "elo_diff",
    "form5_diff", "last5_win_rate", "opp_last5_win_rate",
    "rest_diff",
    "sos", "opp_sos", "sos_diff",
]

new_features = [
    # set dominance
    "cum_set_win_rate", "opp_cum_set_win_rate", "set_win_rate_diff",
    "cum_avg_set_margin", "opp_cum_avg_set_margin", "set_margin_diff",
    # home/away win rates
    "team_home_winrate", "team_away_winrate",
    "opp_home_winrate", "opp_away_winrate",
    # opponent-adjusted hitting
    "adj_hit_pct", "adj_hit_pct_diff",
    # EWA momentum
    "ewa_win_rate", "opp_ewa_win_rate", "ewa_form_diff",
    # head-to-head
    "h2h_win_rate", "opp_h2h_win_rate", "h2h_win_rate_diff",
    # season win rate
    "season_win_rate", "opp_season_win_rate", "season_win_rate_diff",
    # consistency
    "set_margin_std", "opp_set_margin_std", "consistency_diff",
    # cumulative win rate
    "cum_win_rate", "opp_cum_win_rate", "cum_win_rate_diff",
    # per-set stats
    "kills_per_set", "opp_kills_per_set", "kills_per_set_diff",
    "errors_per_set", "opp_errors_per_set", "errors_per_set_diff",
    "aces_per_set", "opp_aces_per_set", "aces_per_set_diff",
    "assists_per_set", "opp_assists_per_set", "assists_per_set_diff",
    "digs_per_set", "opp_digs_per_set", "digs_per_set_diff",
    # rolling hit pct
    "rolling_hit_pct", "opp_rolling_hit_pct", "rolling_hit_pct_diff",
    # streak
    "streak", "opp_streak", "streak_diff",
    # additional rolling form
    "last3_win_rate", "opp_last3_win_rate", "form3_diff",
    "last10_win_rate", "opp_last10_win_rate", "form10_diff",
    # conference & game context
    "conf_avg_elo", "elo_vs_conf", "game_num",
]

features = core_features + new_features

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=core_features + ["win"])

for col in new_features:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

print(f"\nTotal rows: {len(df)}")
print(f"Features: {len(features) + len(loc_cols)}")
wins = df["win"].sum()
losses = (df["win"] == 0).sum()
print(f"Wins: {wins}, Losses: {losses}, Symmetric: {wins == losses}")



# =============================================================================
# ROLLING WALK-FORWARD TRAINING
# =============================================================================
seasons = sorted(df["season"].unique())
all_preds, all_probs, all_true = [], [], []
all_test_indices = []

for i in range(1, len(seasons)):
    train_seasons = seasons[:i]
    test_season   = seasons[i]
    print(f"\nTraining on {train_seasons} → Testing on {test_season}")

    train = df[df["season"].isin(train_seasons)]
    test  = df[df["season"] == test_season]

    if len(train) == 0 or len(test) == 0:
        print("Skipping — empty split")
        continue

    X_train_full = train[features + loc_cols]
    y_train_full = train["win"]
    X_test       = test[features + loc_cols]
    y_test       = test["win"]

    # Split training data into train/validation for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # store results per model
    results = {
        "logreg": {"preds": [], "probs": [], "true": []},
        "rf": {"preds": [], "probs": [], "true": []},
        "gbm": {"preds": [], "probs": [], "true": []},
}

    models = {
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "rf": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "gbm": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
}

    for name, model in models.items():

        model.fit(
            X_tr, y_tr
        )

        probs = model.predict_proba(X_test)[:, 1]

        # threshold tuning (UNCHANGED LOGIC)
        val_probs = model.predict_proba(X_val)[:, 1]

        best_acc, best_t = 0.0, 0.5
        for t in np.arange(0.30, 0.71, 0.01):
            acc = ((val_probs >= t).astype(int) == y_val.values).mean()
            if acc > best_acc:
                best_acc, best_t = acc, t

        preds = (probs >= best_t).astype(int)

        print(f"{name.upper()} → t={best_t:.2f}, val acc={best_acc:.4f}")

        results[name]["probs"].extend(probs)
        results[name]["preds"].extend(preds)
        results[name]["true"].extend(y_test)

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

for name in results:
    print(f"\n{name.upper()}")
    print("-" * 40)

    y_true = results[name]["true"]
    y_pred = results[name]["preds"]
    y_prob = results[name]["probs"]

    print(classification_report(y_true, y_pred))
    print("ROC AUC:", round(roc_auc_score(y_true, y_prob), 4))
