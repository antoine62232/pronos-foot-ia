# ============================================================
# ROLE : Simuler la phase eliminatoire de la CdM 2026 avec le
#        modele de production (forme reelle + Elo) et sauvegarder
#        le bracket complet + le champion.
# ============================================================

import pandas as pd
from joblib import load

PAYS_HOTES_CDM_2026 = ["United States", "Canada", "Mexico"]

FEATURES = [
    "forme_attaque_domicile", "forme_attaque_exterieur",
    "forme_defense_domicile", "forme_defense_exterieur",
    "forme_attaque_domicile_10", "forme_attaque_exterieur_10",
    "forme_defense_domicile_10", "forme_defense_exterieur_10",
    "elo_domicile", "elo_exterieur",
    "match_neutre",
]


def forme_domicile(equipe, historique, nb):
    m = historique[historique["home_team"] == equipe].tail(nb)
    if len(m) == 0:
        return 0.0, 1.0
    return round(m["home_score"].mean(), 2), round(m["away_score"].mean(), 2)


def forme_exterieur(equipe, historique, nb):
    m = historique[historique["away_team"] == equipe].tail(nb)
    if len(m) == 0:
        return 0.0, 1.0
    return round(m["away_score"].mean(), 2), round(m["home_score"].mean(), 2)


# 1. On charge le modele de prod (deja entraine) + l'Elo courant.
print("Chargement du modele + Elo courant...")
modele = load("models/modele_football.pkl")
encoder = load("models/label_encoder.pkl")

historique = pd.read_csv("data/matchs_entrainement.csv")
historique["date"] = pd.to_datetime(historique["date"])
historique = historique.sort_values("date")

df_elo = pd.read_csv("data/elo_actuel.csv")
elos = dict(zip(df_elo["equipe"], df_elo["elo"]))


def simuler_match_couperet(eq1, eq2):
    """Match a elimination directe (terrain neutre). Force un vainqueur."""
    a5, d5 = forme_domicile(eq1, historique, 5)
    ae5, de5 = forme_exterieur(eq2, historique, 5)
    a10, d10 = forme_domicile(eq1, historique, 10)
    ae10, de10 = forme_exterieur(eq2, historique, 10)
    donnees = pd.DataFrame([[a5, ae5, d5, de5, a10, ae10, d10, de10,
                             elos.get(eq1, 1500), elos.get(eq2, 1500), 1]], columns=FEATURES)
    proba = dict(zip(encoder.inverse_transform(modele.classes_), modele.predict_proba(donnees)[0]))
    p1, p2 = proba.get("1", 0), proba.get("2", 0)
    # Pas de nul en phase eliminatoire : on garde l'equipe la plus probable
    if p1 >= p2:
        return eq1, p1 / (p1 + p2) * 100
    return eq2, p2 / (p1 + p2) * 100


# 2. Construction du tableau des 1/16 (structure FIFA)
df_qualifies = pd.read_csv("data/qualifies_1_16.csv")
premiers = df_qualifies[df_qualifies["qualification"].str.startswith("1er")].set_index("groupe")["equipe"].to_dict()
deuxiemes = df_qualifies[df_qualifies["qualification"].str.startswith("2eme")].set_index("groupe")["equipe"].to_dict()
troisiemes = df_qualifies[df_qualifies["mode"] == "repechage"]["equipe"].tolist()

matchs = [
    (deuxiemes["A"], deuxiemes["B"]), (premiers["C"], deuxiemes["F"]),
    (premiers["E"], troisiemes.pop(0)), (premiers["F"], deuxiemes["C"]),
    (deuxiemes["E"], deuxiemes["I"]), (premiers["I"], troisiemes.pop(0)),
    (premiers["A"], troisiemes.pop(0)), (premiers["L"], troisiemes.pop(0)),
    (premiers["G"], troisiemes.pop(0)), (premiers["D"], troisiemes.pop(0)),
    (premiers["H"], deuxiemes["J"]), (deuxiemes["K"], deuxiemes["L"]),
    (premiers["B"], troisiemes.pop(0)), (deuxiemes["D"], deuxiemes["G"]),
    (premiers["J"], deuxiemes["H"]), (premiers["K"], troisiemes.pop(0)),
]

# 3. Simulation tour par tour
noms_tours = ["1/16 de finale", "1/8 de finale", "Quarts de finale", "Demi-finales", "Finale"]
historique_bracket = []
champion = ""

for tour in noms_tours:
    vainqueurs = []
    for eq1, eq2 in matchs:
        gagnant, conf = simuler_match_couperet(eq1, eq2)
        vainqueurs.append(gagnant)
        historique_bracket.append({"tour": tour, "equipe_1": eq1, "equipe_2": eq2,
                                    "vainqueur": gagnant, "confiance": round(conf, 1)})
    if len(vainqueurs) > 1:
        matchs = [(vainqueurs[i], vainqueurs[i + 1]) for i in range(0, len(vainqueurs), 2)]
    else:
        champion = vainqueurs[0]

# 4. Sauvegarde
pd.DataFrame(historique_bracket).to_csv("data/bracket_complet.csv", index=False)
pd.DataFrame([{"role": "Champion", "equipe": champion}]).to_csv("data/vainqueur_final.csv", index=False)
print(f"CHAMPION PREDIT : {champion}")
print("bracket_complet.csv + vainqueur_final.csv sauvegardes.")