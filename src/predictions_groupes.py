# ============================================================
# ROLE : Predire le classement de chaque groupe de la CdM 2026
#        en utilisant le modele de production sauvegarde (forme + Elo).
# ============================================================

import pandas as pd
from joblib import load

PAYS_HOTES_CDM_2026 = ["United States", "Canada", "Mexico"]

GROUPES = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czech Republic'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['United States', 'Australia', 'Paraguay', 'Turkey'],
    'E': ['Germany', 'Ecuador', 'Ivory Coast', 'Curaçao'],
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    'G': ['Belgium', 'Iran', 'Egypt', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cape Verde'],
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'J': ['Argentina', 'Austria', 'Algeria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'Uzbekistan', 'DR Congo'],
    'L': ['England', 'Croatia', 'Panama', 'Ghana'],
}

FEATURES = [
    "forme_attaque_domicile", "forme_attaque_exterieur",
    "forme_defense_domicile", "forme_defense_exterieur",
    "forme_attaque_domicile_10", "forme_attaque_exterieur_10",
    "forme_defense_domicile_10", "forme_defense_exterieur_10",
    "elo_domicile", "elo_exterieur",
    "match_neutre",
]


def forme_domicile(equipe, historique, nb):
    """(buts marques, encaisses) sur les nb derniers matchs A DOMICILE."""
    m = historique[historique["home_team"] == equipe].tail(nb)
    if len(m) == 0:
        return 0.0, 1.0
    return round(m["home_score"].mean(), 2), round(m["away_score"].mean(), 2)


def forme_exterieur(equipe, historique, nb):
    """(buts marques, encaisses) sur les nb derniers matchs A L'EXTERIEUR."""
    m = historique[historique["away_team"] == equipe].tail(nb)
    if len(m) == 0:
        return 0.0, 1.0
    return round(m["away_score"].mean(), 2), round(m["home_score"].mean(), 2)


# 1. On CHARGE le modele de prod (deja entraine) au lieu d'en reentrainer un.
print("Chargement du modele de production + Elo courant...")
modele = load("models/modele_football.pkl")
encoder = load("models/label_encoder.pkl")

historique = pd.read_csv("data/matchs_entrainement.csv")
historique["date"] = pd.to_datetime(historique["date"])
historique = historique.sort_values("date")

df_elo = pd.read_csv("data/elo_actuel.csv")
elos = dict(zip(df_elo["equipe"], df_elo["elo"]))

# 2. Prediction des 72 matchs (forme correcte + Elo)
df_matchs = pd.read_csv("data/matchs_a_predire.csv")

lignes = []
for _, match in df_matchs.iterrows():
    dom, ext = match["home_team"], match["away_team"]
    a5, d5 = forme_domicile(dom, historique, 5)
    ae5, de5 = forme_exterieur(ext, historique, 5)
    a10, d10 = forme_domicile(dom, historique, 10)
    ae10, de10 = forme_exterieur(ext, historique, 10)
    neutre = 0 if dom in PAYS_HOTES_CDM_2026 else 1
    lignes.append([a5, ae5, d5, de5, a10, ae10, d10, de10,
                   elos.get(dom, 1500), elos.get(ext, 1500), neutre])

X = pd.DataFrame(lignes, columns=FEATURES)
proba_df = pd.DataFrame(modele.predict_proba(X), columns=list(encoder.inverse_transform(modele.classes_)))

df_matchs["pronostic"] = proba_df.idxmax(axis=1).values
df_matchs["proba_1"] = proba_df["1"].values
df_matchs["proba_N"] = proba_df["N"].values
df_matchs["proba_2"] = proba_df["2"].values
print(f"{len(df_matchs)} matchs predits avec le modele Elo.")


# 3. Classement de chaque groupe (3 pts victoire, 1 pt nul ; force = somme des probas)
def calculer_classement_groupe(df_groupe, equipes_groupe):
    classement = {eq: {"equipe": eq, "points": 0, "victoires": 0, "nuls": 0,
                       "defaites": 0, "force_predite": 0.0} for eq in equipes_groupe}
    for _, match in df_groupe.iterrows():
        dom, ext, prono = match["home_team"], match["away_team"], match["pronostic"]
        if prono == "1":
            classement[dom]["points"] += 3; classement[dom]["victoires"] += 1; classement[ext]["defaites"] += 1
        elif prono == "2":
            classement[ext]["points"] += 3; classement[ext]["victoires"] += 1; classement[dom]["defaites"] += 1
        else:
            classement[dom]["points"] += 1; classement[ext]["points"] += 1
            classement[dom]["nuls"] += 1; classement[ext]["nuls"] += 1
        classement[dom]["force_predite"] += match["proba_1"]
        classement[ext]["force_predite"] += match["proba_2"]

    df_c = pd.DataFrame(classement.values()).sort_values(
        by=["points", "force_predite"], ascending=[False, False]).reset_index(drop=True)
    df_c["position"] = range(1, len(df_c) + 1)
    return df_c


classements = {lettre: calculer_classement_groupe(df_matchs[df_matchs["group"] == lettre], equipes)
               for lettre, equipes in GROUPES.items()}

# 4. Les 8 meilleurs troisiemes
troisiemes = pd.DataFrame([
    {"groupe": lettre, "equipe": df_c.iloc[2]["equipe"],
     "points": df_c.iloc[2]["points"], "force": df_c.iloc[2]["force_predite"]}
    for lettre, df_c in classements.items()
]).sort_values(by=["points", "force"], ascending=[False, False]).reset_index(drop=True)

# 5. Sauvegarde
df_matchs.to_csv("data/predictions_groupes.csv", index=False)

qualifies = []
for lettre, df_c in classements.items():
    qualifies.append({"qualification": f"1er Groupe {lettre}", "equipe": df_c.iloc[0]["equipe"], "groupe": lettre, "mode": "direct"})
    qualifies.append({"qualification": f"2eme Groupe {lettre}", "equipe": df_c.iloc[1]["equipe"], "groupe": lettre, "mode": "direct"})
for _, row in troisiemes.head(8).iterrows():
    qualifies.append({"qualification": f"3eme Groupe {row['groupe']} (repechage)", "equipe": row["equipe"], "groupe": row["groupe"], "mode": "repechage"})

pd.DataFrame(qualifies).to_csv("data/qualifies_1_16.csv", index=False)
print(f"{len(qualifies)} qualifies sauvegardes (predictions_groupes.csv + qualifies_1_16.csv).")