"""Prediction "live" : re-predit la CdM 2026 en injectant les resultats reels deja joues.

Principe : on repart de l'Elo et de l'historique d'avant-tournoi, on y applique les
matchs deja disputes (data/resultats_reels.csv), puis on relance EXACTEMENT la meme
logique de prediction que predictions_groupes.py / predictions_phase_eliminatoire.py.

Important : on n'ecrase jamais les pronostics geles d'avant-tournoi (sinon l'onglet
Realite VS IA n'aurait plus de sens). On ecrit dans des fichiers separes *_live.csv.
"""

import sys
from pathlib import Path

import pandas as pd
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.elo import maj_elo

PAYS_HOTES = ["United States", "Canada", "Mexico"]

FEATURES = [
    "forme_attaque_domicile", "forme_attaque_exterieur",
    "forme_defense_domicile", "forme_defense_exterieur",
    "forme_attaque_domicile_10", "forme_attaque_exterieur_10",
    "forme_defense_domicile_10", "forme_defense_exterieur_10",
    "elo_domicile", "elo_exterieur", "match_neutre",
]

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

# Le CSV des resultats reels (football-data.org) ecrit certains noms autrement que nos
# pronostics. On inverse donc la correspondance de components/resultats.py (nom CSV -> notre nom).
NOMS_CSV_VERS_NOUS = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
}

modele = load("models/modele_football.pkl")
encoder = load("models/label_encoder.pkl")


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


def _features_match(dom, ext, elos, historique, neutre):
    a5, d5 = forme_domicile(dom, historique, 5)
    ae5, de5 = forme_exterieur(ext, historique, 5)
    a10, d10 = forme_domicile(dom, historique, 10)
    ae10, de10 = forme_exterieur(ext, historique, 10)
    return [a5, ae5, d5, de5, a10, ae10, d10, de10,
            elos.get(dom, 1500), elos.get(ext, 1500), neutre]


def predire_groupes(elos, historique):
    """Predit les 72 matchs de poule (meme logique que predictions_groupes.py)."""
    df = pd.read_csv("data/matchs_a_predire.csv")
    lignes = []
    for _, match in df.iterrows():
        neutre = 0 if match["home_team"] in PAYS_HOTES else 1
        lignes.append(_features_match(match["home_team"], match["away_team"], elos, historique, neutre))
    X = pd.DataFrame(lignes, columns=FEATURES)
    proba = pd.DataFrame(modele.predict_proba(X), columns=list(encoder.inverse_transform(modele.classes_)))
    df["pronostic"] = proba.idxmax(axis=1).values
    df["proba_1"] = proba["1"].values
    df["proba_N"] = proba["N"].values
    df["proba_2"] = proba["2"].values
    return df


def classement_groupe(df_groupe, equipes):
    c = {e: {"equipe": e, "points": 0, "force_predite": 0.0} for e in equipes}
    for _, m in df_groupe.iterrows():
        dom, ext, p = m["home_team"], m["away_team"], m["pronostic"]
        if p == "1":
            c[dom]["points"] += 3
        elif p == "2":
            c[ext]["points"] += 3
        else:
            c[dom]["points"] += 1
            c[ext]["points"] += 1
        c[dom]["force_predite"] += m["proba_1"]
        c[ext]["force_predite"] += m["proba_2"]
    return pd.DataFrame(c.values()).sort_values(["points", "force_predite"], ascending=[False, False]).reset_index(drop=True)


def qualifies(df_predictions):
    classements = {l: classement_groupe(df_predictions[df_predictions["group"] == l], eqs)
                   for l, eqs in GROUPES.items()}
    troisiemes = pd.DataFrame([
        {"groupe": l, "equipe": d.iloc[2]["equipe"], "points": d.iloc[2]["points"], "force": d.iloc[2]["force_predite"]}
        for l, d in classements.items()
    ]).sort_values(["points", "force"], ascending=[False, False]).reset_index(drop=True)
    premiers = {l: d.iloc[0]["equipe"] for l, d in classements.items()}
    deuxiemes = {l: d.iloc[1]["equipe"] for l, d in classements.items()}
    return premiers, deuxiemes, troisiemes.head(8)["equipe"].tolist()


def simuler_match_couperet(eq1, eq2, elos, historique):
    """Match a elimination directe (terrain neutre) : force un vainqueur, pas de nul."""
    X = pd.DataFrame([_features_match(eq1, eq2, elos, historique, 1)], columns=FEATURES)
    proba = dict(zip(encoder.inverse_transform(modele.classes_), modele.predict_proba(X)[0]))
    p1, p2 = proba.get("1", 0), proba.get("2", 0)
    if p1 >= p2:
        return eq1, p1 / (p1 + p2) * 100
    return eq2, p2 / (p1 + p2) * 100


def simuler_bracket(elos, historique):
    """Re-simule tout le tableau et renvoie (bracket, champion, troisieme)."""
    premiers, deuxiemes, troisiemes = qualifies(predire_groupes(elos, historique))
    t = list(troisiemes)
    matchs = [
        (deuxiemes["A"], deuxiemes["B"]), (premiers["C"], deuxiemes["F"]),
        (premiers["E"], t.pop(0)), (premiers["F"], deuxiemes["C"]),
        (deuxiemes["E"], deuxiemes["I"]), (premiers["I"], t.pop(0)),
        (premiers["A"], t.pop(0)), (premiers["L"], t.pop(0)),
        (premiers["G"], t.pop(0)), (premiers["D"], t.pop(0)),
        (premiers["H"], deuxiemes["J"]), (deuxiemes["K"], deuxiemes["L"]),
        (premiers["B"], t.pop(0)), (deuxiemes["D"], deuxiemes["G"]),
        (premiers["J"], deuxiemes["H"]), (premiers["K"], t.pop(0)),
    ]
    bracket, champion, troisieme, perdants_demis = [], "", "", []
    for tour in ["1/16 de finale", "1/8 de finale", "Quarts de finale", "Demi-finales", "Finale"]:
        vainqueurs = []
        for eq1, eq2 in matchs:
            gagnant, conf = simuler_match_couperet(eq1, eq2, elos, historique)
            vainqueurs.append(gagnant)
            bracket.append({"tour": tour, "equipe_1": eq1, "equipe_2": eq2,
                            "vainqueur": gagnant, "confiance": round(conf, 1)})
            if tour == "Demi-finales":
                perdants_demis.append(eq2 if gagnant == eq1 else eq1)
        if len(vainqueurs) > 1:
            matchs = [(vainqueurs[i], vainqueurs[i + 1]) for i in range(0, len(vainqueurs), 2)]
        else:
            champion = vainqueurs[0]
    if len(perdants_demis) == 2:
        troisieme, conf3 = simuler_match_couperet(perdants_demis[0], perdants_demis[1], elos, historique)
        bracket.append({"tour": "Petite finale", "equipe_1": perdants_demis[0],
                        "equipe_2": perdants_demis[1], "vainqueur": troisieme, "confiance": round(conf3, 1)})
    return pd.DataFrame(bracket), champion, troisieme


def resultats_reels_format_nous():
    """Lit data/resultats_reels.csv et le renvoie au format de notre historique
    (nos noms d'equipes, orientation domicile/exterieur de nos fixtures, drapeau neutre)."""
    fixtures = pd.read_csv("data/matchs_a_predire.csv")
    par_paire = {frozenset({r["home_team"], r["away_team"]}): r for _, r in fixtures.iterrows()}

    reel = pd.read_csv("data/resultats_reels.csv")
    lignes = []
    for _, m in reel.iterrows():
        # Un match sans score final (en cours, reporte, donnee manquante) n'est pas
        # exploitable pour l'Elo : on l'ignore au lieu de planter sur int(NaN).
        if pd.isna(m["buts_domicile"]) or pd.isna(m["buts_exterieur"]):
            continue
        dom = NOMS_CSV_VERS_NOUS.get(m["equipe_domicile"], m["equipe_domicile"])
        ext = NOMS_CSV_VERS_NOUS.get(m["equipe_exterieur"], m["equipe_exterieur"])
        fix = par_paire.get(frozenset({dom, ext}))
        if fix is None:
            print(f"  [ignore] match introuvable dans les fixtures : {dom} vs {ext}")
            continue
        # On reoriente le score selon NOTRE domicile/exterieur (le CSV peut les inverser).
        if fix["home_team"] == dom:
            sd, se = m["buts_domicile"], m["buts_exterieur"]
        else:
            sd, se = m["buts_exterieur"], m["buts_domicile"]
        lignes.append({"date": fix["date"], "home_team": fix["home_team"], "away_team": fix["away_team"],
                       "home_score": int(sd), "away_score": int(se), "neutral": fix["neutral"]})
    return pd.DataFrame(lignes)


def main():
    # Etat gele d'avant-tournoi
    historique = pd.read_csv("data/matchs_entrainement.csv")
    historique["date"] = pd.to_datetime(historique["date"])
    historique = historique.sort_values("date")
    df_elo = pd.read_csv("data/elo_actuel.csv")
    elos_geles = dict(zip(df_elo["equipe"], df_elo["elo"]))

    # Resultats reels -> mise a jour de l'Elo et de l'historique de forme
    wc = resultats_reels_format_nous()
    print(f"{len(wc)} resultats reels injectes.")

    elos_live = dict(elos_geles)
    for _, m in wc.sort_values("date").iterrows():
        nd, na = maj_elo(elos_live.get(m["home_team"], 1500), elos_live.get(m["away_team"], 1500),
                         m["home_score"], m["away_score"], bool(m["neutral"]))
        elos_live[m["home_team"]] = nd
        elos_live[m["away_team"]] = na

    wc["date"] = pd.to_datetime(wc["date"])
    historique_live = pd.concat([historique, wc], ignore_index=True).sort_values("date")

    # Predictions live
    pred_live = predire_groupes(elos_live, historique_live)
    bracket_live, champion, troisieme = simuler_bracket(elos_live, historique_live)

    # Sauvegarde dans des fichiers SEPARES (on ne touche pas aux fichiers geles)
    pred_live.to_csv("data/predictions_groupes_live.csv", index=False)
    bracket_live.to_csv("data/bracket_complet_live.csv", index=False)
    pd.DataFrame([{"role": "Champion", "equipe": champion},
                  {"role": "3e place", "equipe": troisieme}]).to_csv("data/vainqueur_final_live.csv", index=False)

    print(f"CHAMPION LIVE : {champion} (3e : {troisieme})")
    print("Ecrits : predictions_groupes_live.csv, bracket_complet_live.csv, vainqueur_final_live.csv")


if __name__ == "__main__":
    main()
