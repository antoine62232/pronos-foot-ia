"""Backtest du modele V2 sur l'ensemble des matchs de competition.

Separation temporelle stricte : entrainement avant DATE_CUTOFF, test apres.
Les features sont calculees une seule fois (rolling + shift) afin de rester
identiques entre l'entrainement et la prediction.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from metriques import evaluer
from backtest import POINTS_FIFA

DATE_CUTOFF = "2018-01-01"


def determiner_resultat(ligne):
    if ligne["home_score"] > ligne["away_score"]:
        return "1"
    if ligne["home_score"] == ligne["away_score"]:
        return "N"
    return "2"


# ============================================================
# 1. Chargement
# ============================================================
df = pd.read_csv("data/matchs_entrainement.csv")
df["date"] = pd.to_datetime(df["date"])
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
df = df.sort_values(by="date").reset_index(drop=True)


# ============================================================
# 2. Features (shift(1) exclut le match courant du calcul de la forme)
# ============================================================
for n in (5, 10):
    suf = "" if n == 5 else "_10"
    df[f"forme_attaque_domicile{suf}"] = df.groupby("home_team")["home_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_attaque_exterieur{suf}"] = df.groupby("away_team")["away_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_defense_domicile{suf}"] = df.groupby("home_team")["away_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_defense_exterieur{suf}"] = df.groupby("away_team")["home_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))

df["points_fifa_domicile"] = df["home_team"].map(POINTS_FIFA).fillna(1200)
df["points_fifa_exterieur"] = df["away_team"].map(POINTS_FIFA).fillna(1200)
df["match_neutre"] = df["neutral"].astype(int)
df["resultat"] = df.apply(determiner_resultat, axis=1)
df = df.fillna(0)

features = [
    "forme_attaque_domicile", "forme_attaque_exterieur",
    "forme_defense_domicile", "forme_defense_exterieur",
    "forme_attaque_domicile_10", "forme_attaque_exterieur_10",
    "forme_defense_domicile_10", "forme_defense_exterieur_10",
    "points_fifa_domicile", "points_fifa_exterieur",
    "match_neutre",
]


# ============================================================
# 3. Separation temporelle
# ============================================================
cutoff = pd.to_datetime(DATE_CUTOFF)
df_train = df[df["date"] < cutoff]
df_test = df[(df["date"] >= cutoff) & (df["tournament"] != "Friendly")]
print(f"Train : {len(df_train)} matchs (avant {cutoff.date()})")
print(f"Test  : {len(df_test)} matchs de competition (apres {cutoff.date()})")


# ============================================================
# 4. Entrainement
# ============================================================
encoder = LabelEncoder()
y_train = encoder.fit_transform(df_train["resultat"])
poids = compute_sample_weight(class_weight="balanced", y=y_train)

modele = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
modele.fit(df_train[features], y_train, sample_weight=poids)


# ============================================================
# 5. Prediction (colonnes remises dans l'ordre domicile / nul / exterieur)
# ============================================================
labels_ordre = encoder.inverse_transform(modele.classes_)
proba_df = pd.DataFrame(modele.predict_proba(df_test[features]), columns=labels_ordre)
y_proba = proba_df[["1", "N", "2"]].values
y_true = df_test["resultat"].map({"1": 0, "N": 1, "2": 2}).values


# ============================================================
# 6. Resultats globaux
# ============================================================
print("\nVerdict global :")
evaluer(y_true, y_proba)


# ============================================================
# 7. Resultats par type de competition
# ============================================================
def categorie(t):
    return "Qualifications" if "qualification" in t else "Phases finales / autres"

cat = df_test["tournament"].map(categorie).values
for nom in ["Phases finales / autres", "Qualifications"]:
    masque = cat == nom
    if masque.sum() == 0:
        continue
    print(f"\n{nom} ({masque.sum()} matchs) :")
    evaluer(y_true[masque], y_proba[masque])


# ============================================================
# 8. Calibration
# ============================================================
confiance = y_proba.max(axis=1)
predit = y_proba.argmax(axis=1)
correct = (predit == y_true).astype(int)

tranches = [0.33, 0.40, 0.50, 0.60, 0.70, 0.80, 1.001]
labels = ["33-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
print("\nCalibration :")
print(f"{'Confiance':>12} | {'Nb':>5} | {'Moyenne':>8} | {'Reel':>6}")
for i in range(len(labels)):
    m = (confiance >= tranches[i]) & (confiance < tranches[i + 1])
    if m.sum() == 0:
        continue
    print(f"{labels[i]:>12} | {m.sum():>5} | {confiance[m].mean():>7.0%} | {correct[m].mean():>5.0%}")