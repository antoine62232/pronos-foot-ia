"""Compare plusieurs jeux de features pour mesurer l'apport de l'Elo."""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from metriques import evaluer
from elo import calculer_elo
from backtest import POINTS_FIFA

DATE_CUTOFF = "2018-01-01"

# Test : la valeur marchande n'ameliore pas le RPS par-dessus l'Elo (info redondante).
# Mesure sur 2741 matchs de phases finales : RPS 0.197 inchange. Hypothese rejetee
MARKET_VALUE = {
    "England": 1345, "France": 1195, "Brazil": 1135, "Portugal": 1000,
    "Spain": 861, "Argentina": 821, "Germany": 775, "Netherlands": 672,
    "Belgium": 549, "Uruguay": 424, "Denmark": 347, "Croatia": 326,
    "Morocco": 318, "Serbia": 292, "Japan": 285, "Switzerland": 282,
    "United States": 270, "Poland": 254, "Ghana": 242, "Ecuador": 236,
    "Senegal": 212, "Canada": 185, "South Korea": 184, "Cameroon": 176,
    "Wales": 175, "Mexico": 165, "Tunisia": 54, "Iran": 51,
    "Australia": 41, "Saudi Arabia": 15, "Qatar": 14, "Costa Rica": 12,
}


def determiner_resultat(ligne):
    if ligne["home_score"] > ligne["away_score"]:
        return "1"
    if ligne["home_score"] == ligne["away_score"]:
        return "N"
    return "2"


def entrainer_et_evaluer(df_train, df_test, features, nom):
    """Entraine un modele avec un jeu de features donne et affiche ses metriques."""
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df_train["resultat"])
    poids = compute_sample_weight(class_weight="balanced", y=y_train)
    modele = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
    modele.fit(df_train[features], y_train, sample_weight=poids)

    labels_ordre = encoder.inverse_transform(modele.classes_)
    proba_df = pd.DataFrame(modele.predict_proba(df_test[features]), columns=labels_ordre)
    y_proba = proba_df[["1", "N", "2"]].values
    y_true = df_test["resultat"].map({"1": 0, "N": 1, "2": 2}).values

    r = evaluer(y_true, y_proba, afficher=False)
    print(f"{nom:26} | acc {r['accuracy']:.3f} | logloss {r['log_loss']:.3f} | brier {r['brier']:.3f} | rps {r['rps']:.3f}")
    return r


# ============================================================
# 1. Chargement + Elo + features de forme
# ============================================================
df = pd.read_csv("data/matchs_entrainement.csv")
df["date"] = pd.to_datetime(df["date"])
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
df = calculer_elo(df)   # trie par date et ajoute elo_domicile / elo_exterieur

for n in (5, 10):
    suf = "" if n == 5 else "_10"
    df[f"forme_attaque_domicile{suf}"] = df.groupby("home_team")["home_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_attaque_exterieur{suf}"] = df.groupby("away_team")["away_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_defense_domicile{suf}"] = df.groupby("home_team")["away_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_defense_exterieur{suf}"] = df.groupby("away_team")["home_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))

df["points_fifa_domicile"] = df["home_team"].map(POINTS_FIFA).fillna(1200)
df["points_fifa_exterieur"] = df["away_team"].map(POINTS_FIFA).fillna(1200)
# Valeur marchande dom/ext. Defaut bas (20 M€) pour les equipes hors dico.
df["valeur_domicile"] = df["home_team"].map(MARKET_VALUE).fillna(20)
df["valeur_exterieur"] = df["away_team"].map(MARKET_VALUE).fillna(20)
df["match_neutre"] = df["neutral"].astype(int)
df["resultat"] = df.apply(determiner_resultat, axis=1)
df = df.fillna(0)

# ============================================================
# 2. Les trois jeux de features a comparer
# ============================================================
forme = [
    "forme_attaque_domicile", "forme_attaque_exterieur",
    "forme_defense_domicile", "forme_defense_exterieur",
    "forme_attaque_domicile_10", "forme_attaque_exterieur_10",
    "forme_defense_domicile_10", "forme_defense_exterieur_10",
]
fifa = ["points_fifa_domicile", "points_fifa_exterieur"]
elo = ["elo_domicile", "elo_exterieur"]

valeur = ["valeur_domicile", "valeur_exterieur"]

configs = {
    "Forme + Elo (prod actuel)": forme + elo + ["match_neutre"],
    "Forme + Elo + Valeur":       forme + elo + valeur + ["match_neutre"],
}
# ============================================================
# 3. Separation temporelle + comparaison
# ============================================================
cutoff = pd.to_datetime(DATE_CUTOFF)
df_train = df[df["date"] < cutoff]
df_test = df[(df["date"] >= cutoff) & (df["tournament"] != "Friendly")]
finales = df_test[~df_test["tournament"].str.contains("qualification")]

print(f"\n=== PHASES FINALES uniquement ({len(finales)} matchs) ===")
for nom, feats in configs.items():
    entrainer_et_evaluer(df_train, finales, feats, nom)

print(f"\n=== TOUTES competitions ({len(df_test)} matchs) ===")
for nom, feats in configs.items():
    entrainer_et_evaluer(df_train, df_test, feats, nom)