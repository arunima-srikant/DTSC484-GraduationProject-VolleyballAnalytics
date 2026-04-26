
import re
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches



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

# Calendar week relative to each season start (for weekly accuracy plots)
season_col = "season" if "season" in df.columns else "Season"
df["season_start"] = df.groupby(season_col)["date"].transform("min")
df["week_of_season"] = ((df["date"] - df["season_start"]).dt.days // 7) + 1

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

df.to_csv("SEM8/Grad Project/team_match_features.csv", index=False)

def assign_confidence(prob):
    distance = abs(prob - 0.5)
    if distance >= 0.15:   return "High"
    elif distance >= 0.08: return "Medium"
    else:                  return "Low"

def confidence_color(tier):
    return {"High": "green", "Medium": "orange", "Low": "red"}[tier]


# =============================================================================
# 9. WALK-FORWARD XGBOOST TRAINING
# =============================================================================
seasons = sorted(df["season"].unique())
all_preds, all_probs, all_true = [], [], []
all_teams, all_opps, all_dates, all_seasons, all_weeks = [], [], [], [], []

for i in range(1, len(seasons)):
    train_seasons = seasons[:i]
    test_season = seasons[i]
    print(f"\nTraining on {train_seasons} → Testing on {test_season}")

    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season]

    if len(train) == 0 or len(test) == 0:
        print("Skipping — empty split")
        continue

    X_train_full = train[features + loc_cols]
    y_train_full = train["win"]
    X_test = test[features + loc_cols]
    y_test = test["win"]

    # Split training into train/validation for early stopping + threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42
    )

    # Best tuned params
    base_model = XGBClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    base_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # ── Isotonic calibration ───────────────────────────────────────────────
    # XGBoost probabilities are often slightly overconfident (pushed toward
    # 0 and 1). Isotonic regression maps raw probs → calibrated probs that
    # better reflect true win rates. Trained on val set (no leakage).
    # cv="prefit" means: model is already trained, just fit the calibrator.
    calibrated_model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
    calibrated_model.fit(X_val, y_val)

    probs = calibrated_model.predict_proba(X_test)[:, 1]

    # Optimal threshold on val set
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    best_acc, best_t = 0.0, 0.5
    for t in np.arange(0.35, 0.66, 0.01):
        acc = ((val_probs >= t).astype(int) == y_val.values).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"  Threshold: {best_t:.2f}  |  Val acc: {best_acc:.4f}")

    preds = (probs >= best_t).astype(int)

    all_probs.extend(probs)
    all_preds.extend(preds)
    all_true.extend(y_test)
    all_teams.extend(test["team"].values)
    all_opps.extend(test["opp_clean"].values)
    all_dates.extend(test["date"].values)
    all_seasons.extend([test_season] * len(test))
    all_weeks.extend(test["week_of_season"].values)


# =============================================================================
# BUILD RESULTS DATAFRAME WITH CONFIDENCE
# =============================================================================
results = pd.DataFrame({
    "season":      all_seasons,
    "date":        all_dates,
    "week_of_season": all_weeks,
    "team":        all_teams,
    "opponent":    all_opps,
    "actual_win":  all_true,
    "predicted_win": all_preds,
    "win_prob":    np.round(all_probs, 4),
})

results["confidence_tier"] = results["win_prob"].apply(assign_confidence)
results["confidence"] = (results["win_prob"] - 0.5).abs()
results["correct"]         = (results["actual_win"] == results["predicted_win"]).astype(int)
overall_acc = accuracy_score(all_true, all_preds)

# Human-readable prediction label
results["prediction"] = results.apply(
    lambda r: f"{r['team']} wins ({r['win_prob']*100:.1f}%)" if r["predicted_win"] == 1
              else f"{r['opponent']} wins ({(1-r['win_prob'])*100:.1f}%)",
    axis=1
)


# =============================================================================
# OVERALL RESULTS
# =============================================================================
print("\n" + "="*55)
print("OVERALL RESULTS")
print("="*55 + "\n")
print(classification_report(all_true, all_preds))
print("ROC AUC:", round(roc_auc_score(all_true, all_probs), 4))


# =============================================================================
# CONFIDENCE TIER ANALYSIS
# The key question: does HIGH confidence actually mean higher accuracy?
# If the model is well-calibrated, High tier should outperform Medium > Low.
# =============================================================================
print("\n" + "="*55)
print("ACCURACY BY CONFIDENCE TIER")
print("="*55)

tier_stats = (
    results.groupby("confidence_tier")
    .agg(
        games      = ("correct", "count"),
        accuracy   = ("correct", "mean"),
        avg_prob   = ("win_prob", "mean"),
    )
    .reindex(["High", "Medium", "Low"])
)
tier_stats["accuracy_pct"] = (tier_stats["accuracy"] * 100).round(1)
tier_stats["avg_prob_pct"] = (tier_stats["avg_prob"]  * 100).round(1)
print(tier_stats[["games", "accuracy_pct", "avg_prob_pct"]].to_string())


# =============================================================================
# WEEKLY ANALYSIS (averaged across all test seasons)
# =============================================================================
def safe_auc(grp):
    if grp["actual_win"].nunique() < 2 or len(grp) < 5:
        return np.nan
    return roc_auc_score(grp["actual_win"], grp["win_prob"])


by_week = results.groupby("week_of_season").agg(
    n_matches=("correct", "size"),
    accuracy=("correct", "mean"),
).reset_index()

auc_by_week = (
    results.groupby("week_of_season")
    .apply(safe_auc, include_groups=False)
    .reset_index()
)
auc_by_week.columns = ["week_of_season", "roc_auc"]
by_week = by_week.merge(auc_by_week, on="week_of_season", how="left")


# =============================================================================
# CALIBRATION ANALYSIS
# How well do predicted probabilities match actual win rates?
# Perfect calibration: if model says 70% confidence → wins 70% of the time.
# =============================================================================
print("\n" + "="*55)
print("CALIBRATION: PREDICTED PROB vs ACTUAL WIN RATE")
print("="*55)

prob_bins = pd.cut(results["win_prob"], bins=np.arange(0, 1.05, 0.1), include_lowest=True)
calib_table = (
    results.groupby(prob_bins, observed=False)
    .agg(games=("actual_win","count"), actual_win_rate=("actual_win","mean"))
    .reset_index()
)
calib_table["bin_midpoint"]     = np.arange(0.05, 1.05, 0.1)
calib_table["actual_win_rate"]  = calib_table["actual_win_rate"].round(3)
calib_table["gap"]              = (calib_table["bin_midpoint"] - calib_table["actual_win_rate"]).round(3)
print(calib_table[["win_prob","games","bin_midpoint","actual_win_rate","gap"]].to_string(index=False))
print("\nGap = predicted - actual. Positive = overconfident, Negative = underconfident.")

# Confidence bins used by legacy combined confidence chart
conf_bins = [0.0, 0.08, 0.15, 0.50]
conf_labels = ["Low (50–58%)", "Medium (58–65%)", "High (65%+)"]
results["conf_bin"] = pd.cut(
    results["confidence"], bins=conf_bins, labels=conf_labels, include_lowest=True
)
by_tier = results.groupby("conf_bin", observed=True).agg(
    n=("correct", "size"),
    accuracy=("correct", "mean"),
    avg_conf=("confidence", "mean"),
    avg_prob=("win_prob", "mean"),
).reset_index()


# =============================================================================
# SAMPLE PREDICTIONS WITH CONFIDENCE
# =============================================================================
print("\n" + "="*55)
print("SAMPLE PREDICTIONS WITH CONFIDENCE TIERS")
print("="*55)

for tier in ["High", "Medium", "Low"]:
    sample = results[results["confidence_tier"] == tier].sample(
        min(5, len(results[results["confidence_tier"] == tier])), random_state=42
    )
    print(f"\n── {tier} Confidence ──")
    for _, row in sample.iterrows():
        result_str = "✓" if row["correct"] else "✗"
        print(f"  {result_str}  {row['team']:30s} vs {row['opponent']:30s}  |  {row['prediction']}")


# =============================================================================
# VISUALIZATIONS
# =============================================================================
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.30, height_ratios=[1, 1, 1])

# ── Panel 1 (top, full width): Weekly accuracy ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
x_pos = range(len(by_week))
bar_colors = ["#4C72B0" if acc >= overall_acc else "#C44E52" for acc in by_week["accuracy"]]
ax1.bar(x_pos, by_week["accuracy"], color=bar_colors, alpha=0.85, width=0.7,
        edgecolor="white", linewidth=0.4)
ax1.axhline(overall_acc, color="black", ls="--", lw=1.3,
            label=f"Overall accuracy ({overall_acc:.3f})")
wk_labels = [f"Week {int(r['week_of_season'])}" for _, r in by_week.iterrows()]
ax1.set_xticks(x_pos)
ax1.set_xticklabels(wk_labels, fontsize=9, ha="center")
for i, row in enumerate(by_week.itertuples()):
    ax1.text(i, row.accuracy + 0.012, f"n={row.n_matches}",
             ha="center", fontsize=8, color="#555555")
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.set_title("Prediction Accuracy by Week of Season  (XGBoost + Isotonic Calibration)",
              fontsize=13, fontweight="bold")
ax1.set_ylim(0.5, 1.05)
ax1.legend(handles=[
    mpatches.Patch(color="#4C72B0", alpha=0.85, label="≥ overall avg"),
    mpatches.Patch(color="#C44E52", alpha=0.85, label="< overall avg"),
    plt.Line2D([0], [0], color="black", ls="--", lw=1.3, label=f"Overall ({overall_acc:.3f})"),
], fontsize=9, loc="lower right")

# ── Panel 2: Accuracy by confidence tier ─────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
tier_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
tier_labels = [str(t) for t in by_tier["conf_bin"]]
tier_accs = by_tier["accuracy"].values
tier_ns = by_tier["n"].values
ax2.bar(range(len(by_tier)), tier_accs, color=tier_colors, alpha=0.85,
        width=0.55, edgecolor="white", linewidth=0.4)
for i, (acc, n) in enumerate(zip(tier_accs, tier_ns)):
    ax2.text(i, acc + 0.01, f"{acc*100:.1f}%\n(n={n:,})", ha="center", fontsize=9, color="#333333")
ax2.axhline(overall_acc, color="gray", ls="--", lw=1.1, alpha=0.7,
            label=f"Overall ({overall_acc:.3f})")
ax2.set_xticks(range(len(by_tier)))
ax2.set_xticklabels(tier_labels, fontsize=9)
ax2.set_ylabel("Accuracy", fontsize=11)
ax2.set_title("Accuracy by Confidence Tier", fontsize=12, fontweight="bold")
ax2.set_ylim(0.5, 1.05)
ax2.legend(fontsize=9)

# ── Panel 3: Distribution of confidence scores ───────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(results["confidence"], bins=35, color="#4C72B0", alpha=0.80,
         edgecolor="white", linewidth=0.3)
ax3.axvline(0.08, color="#f39c12", ls="--", lw=1.3, label="Low/Med boundary (0.08)")
ax3.axvline(0.15, color="#2ecc71", ls="--", lw=1.3, label="Med/High boundary (0.15)")
ax3.axvline(results["confidence"].median(), color="red", ls=":", lw=1.2,
            label=f"Median ({results['confidence'].median():.2f})")
ax3.set_xlabel("Confidence (|prob - 0.5|)", fontsize=11)
ax3.set_ylabel("Number of predictions", fontsize=11)
ax3.set_title("Distribution of Prediction Confidence", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)

# ── Panel 4: Calibration curve ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
fraction_of_positives, mean_predicted = calibration_curve(all_true, all_probs, n_bins=10, strategy="uniform")
ax4.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
ax4.plot(mean_predicted, fraction_of_positives, "o-", color="#9b59b6", lw=2,
         markersize=7, label="XGBoost (calibrated)", zorder=5)
ax4.fill_between(mean_predicted, fraction_of_positives, mean_predicted,
                 alpha=0.10, color="#9b59b6")
ax4.set_xlabel("Mean Predicted Probability", fontsize=11)
ax4.set_ylabel("Actual Win Rate", fontsize=11)
ax4.set_title("Calibration Curve", fontsize=12, fontweight="bold")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_aspect("equal")
ax4.legend(fontsize=9)

# ── Panel 5: Accuracy vs confidence (smoothed) ───────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
sorted_res = results.sort_values("confidence")
window = min(200, max(50, len(sorted_res) // 10))
rolling_acc = sorted_res["correct"].rolling(window, center=True, min_periods=50).mean()
ax5.plot(sorted_res["confidence"].values, rolling_acc.values,
         color="#4C72B0", lw=2.2, label=f"Rolling accuracy (window={window})")
ax5.fill_between(sorted_res["confidence"].values, rolling_acc.values,
                 alpha=0.15, color="#4C72B0")
ax5.axvline(0.08, color="#f39c12", ls="--", lw=1.1, alpha=0.8)
ax5.axvline(0.15, color="#2ecc71", ls="--", lw=1.1, alpha=0.8)
ax5.axhline(overall_acc, color="gray", ls=":", lw=1, alpha=0.7,
            label=f"Overall accuracy ({overall_acc:.3f})")
ax5.set_xlabel("Confidence (|prob - 0.5|)", fontsize=11)
ax5.set_ylabel(f"Rolling Accuracy (window={window})", fontsize=11)
ax5.set_title("Accuracy vs Confidence (Smoothed)", fontsize=12, fontweight="bold")
ax5.set_ylim(0.4, 1.05)
ax5.legend(fontsize=9)

plt.savefig("SEM8/Grad Project/new_confidence_analysis.png", dpi=150, bbox_inches="tight")
print("\nChart saved → SEM8/Grad Project/new_confidence_analysis.png")
plt.close(fig)


# =============================================================================
# PREDICTION FUNCTION — USE THIS TO PREDICT ANY NEW MATCHUP
# Pass in pre-computed feature values for a new game and get back
# a prediction with confidence tier and calibrated probability.
# =============================================================================
def predict_with_confidence(model, feature_values: dict, feature_cols: list, threshold: float = 0.5):
    """
    Predict outcome + confidence for a single matchup.

    Parameters
    ----------
    model         : fitted CalibratedClassifierCV
    feature_values: dict of {feature_name: value} for the matchup
    feature_cols  : list of feature names in the exact order the model expects
    threshold     : decision threshold (use the optimal one from your last fold)

    Returns
    -------
    dict with prediction, win_probability, confidence_tier, and interpretation
    """
    row = pd.DataFrame([feature_values])[feature_cols].fillna(0)
    prob  = model.predict_proba(row)[0, 1]
    pred  = int(prob >= threshold)
    tier  = assign_confidence(prob)

    tier_descriptions = {
        "High":   "Model is confident — historical stats strongly favour one side.",
        "Medium": "Model leans one way but outcome is less certain.",
        "Low":    "Coin-flip zone — teams are evenly matched by available stats.",
    }

    return {
        "predicted_winner":  "Team wins"     if pred == 1 else "Opponent wins",
        "win_probability":   f"{prob*100:.1f}%",
        "loss_probability":  f"{(1-prob)*100:.1f}%",
        "confidence_tier":   tier,
        "interpretation":    tier_descriptions[tier],
    }


# =============================================================================
# CONFIDENCE SUMMARY — good for a project slide or report
# =============================================================================
print("\n" + "="*55)
print("CONFIDENCE SUMMARY")
print("="*55)

total = len(results)
for tier, color in [("High","High"), ("Medium","Medium"), ("Low","Low")]:
    subset = results[results["confidence_tier"] == tier]
    if len(subset) == 0:
        continue
    pct  = len(subset) / total * 100
    acc  = subset["correct"].mean() * 100
    print(f"\n{tier} Confidence  ({len(subset):,} games, {pct:.1f}% of all predictions)")
    print(f"  Accuracy : {acc:.1f}%")
    print(f"  Avg prob : {subset['win_prob'].mean()*100:.1f}%")
    print(f"  Correct  : {subset['correct'].sum():,}  |  Wrong: {(~subset['correct'].astype(bool)).sum():,}")

print(f"\nOverall: {results['correct'].mean()*100:.1f}% across {total:,} predictions")
print("\nResults dataframe saved to: results_with_confidence_new.csv")
results.to_csv("SEM8/Grad Project/results_with_confidence_new.csv", index=False)