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
from src.bracket_fifa import construire_seiziemes, simuler_tableau

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
    c = {e: {"equipe": e, "points": 0, "bp": 0, "bc": 0, "force_predite": 0.0} for e in equipes}
    for _, m in df_groupe.iterrows():
        dom, ext, p = m["home_team"], m["away_team"], m["pronostic"]
        if p == "1":
            c[dom]["points"] += 3
        elif p == "2":
            c[ext]["points"] += 3
        else:
            c[dom]["points"] += 1
            c[ext]["points"] += 1
        bd = m["buts_dom"] if "buts_dom" in m else pd.NA
        be = m["buts_ext"] if "buts_ext" in m else pd.NA
        if pd.notna(bd) and pd.notna(be):
            c[dom]["bp"] += bd; c[dom]["bc"] += be
            c[ext]["bp"] += be; c[ext]["bc"] += bd
        c[dom]["force_predite"] += m["proba_1"]
        c[ext]["force_predite"] += m["proba_2"]
    df = pd.DataFrame(c.values())
    df["dif"] = df["bp"] - df["bc"]
    # ordre FIFA : points, difference de buts, buts marques
    return df.sort_values(["points", "dif", "bp", "force_predite"], ascending=False).reset_index(drop=True)


def qualifies(df_predictions):
    classements = {l: classement_groupe(df_predictions[df_predictions["group"] == l], eqs)
                   for l, eqs in GROUPES.items()}
    troisiemes = pd.DataFrame([
        {"groupe": l, "equipe": d.iloc[2]["equipe"], "points": d.iloc[2]["points"],
         "dif": d.iloc[2]["dif"], "bp": d.iloc[2]["bp"], "force": d.iloc[2]["force_predite"]}
        for l, d in classements.items()
    ]).sort_values(["points", "dif", "bp", "force"], ascending=False).reset_index(drop=True)
    premiers = {l: d.iloc[0]["equipe"] for l, d in classements.items()}
    deuxiemes = {l: d.iloc[1]["equipe"] for l, d in classements.items()}
    troisiemes_par_groupe = {r["groupe"]: r["equipe"] for _, r in troisiemes.head(8).iterrows()}
    return premiers, deuxiemes, troisiemes_par_groupe


def qualifies_df(df_predictions):
    """Construit le tableau des qualifies au format de qualifies_1_16.csv
    (colonnes qualification, equipe, groupe, mode), pour la version live."""
    classements = {l: classement_groupe(df_predictions[df_predictions["group"] == l], eqs)
                   for l, eqs in GROUPES.items()}
    troisiemes = pd.DataFrame([
        {"groupe": l, "equipe": d.iloc[2]["equipe"], "points": d.iloc[2]["points"],
         "dif": d.iloc[2]["dif"], "bp": d.iloc[2]["bp"], "force": d.iloc[2]["force_predite"]}
        for l, d in classements.items()
    ]).sort_values(["points", "dif", "bp", "force"], ascending=False).reset_index(drop=True)
    lignes = []
    for l, d in classements.items():
        lignes.append({"qualification": f"1er Groupe {l}", "equipe": d.iloc[0]["equipe"], "groupe": l, "mode": "direct"})
        lignes.append({"qualification": f"2eme Groupe {l}", "equipe": d.iloc[1]["equipe"], "groupe": l, "mode": "direct"})
    for _, r in troisiemes.head(8).iterrows():
        lignes.append({"qualification": f"3eme Groupe {r['groupe']} (repechage)",
                       "equipe": r["equipe"], "groupe": r["groupe"], "mode": "repechage"})
    return pd.DataFrame(lignes)


def simuler_match_couperet(eq1, eq2, elos, historique):
    """Match a elimination directe (terrain neutre) : force un vainqueur, pas de nul."""
    X = pd.DataFrame([_features_match(eq1, eq2, elos, historique, 1)], columns=FEATURES)
    proba = dict(zip(encoder.inverse_transform(modele.classes_), modele.predict_proba(X)[0]))
    p1, p2 = proba.get("1", 0), proba.get("2", 0)
    if p1 >= p2:
        return eq1, p1 / (p1 + p2) * 100
    return eq2, p2 / (p1 + p2) * 100


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


def appliquer_resultats_reels(pred, wc):
    reel = {frozenset({m["home_team"], m["away_team"]}): m for _, m in wc.iterrows()}
    pred = pred.copy()
    pred["buts_dom"] = pd.NA
    pred["buts_ext"] = pd.NA
    for i, p in pred.iterrows():
        m = reel.get(frozenset({p["home_team"], p["away_team"]}))
        if m is None:
            continue
        if m["home_team"] == p["home_team"]:
            sd, se = m["home_score"], m["away_score"]
        else:
            sd, se = m["away_score"], m["home_score"]
        iss = "1" if sd > se else ("N" if sd == se else "2")
        pred.at[i, "pronostic"] = iss
        pred.at[i, "proba_1"] = 1.0 if iss == "1" else 0.0
        pred.at[i, "proba_N"] = 1.0 if iss == "N" else 0.0
        pred.at[i, "proba_2"] = 1.0 if iss == "2" else 0.0
        pred.at[i, "buts_dom"] = sd
        pred.at[i, "buts_ext"] = se
    return pred


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

    # Predictions live, puis on FIGE les matchs deja joues sur leur vrai resultat.
    pred_live = predire_groupes(elos_live, historique_live)
    pred_live = appliquer_resultats_reels(pred_live, wc)
    premiers, deuxiemes, troisiemes_par_groupe = qualifies(pred_live)
    seiziemes = construire_seiziemes(premiers, deuxiemes, troisiemes_par_groupe)
    sim = lambda a, b: simuler_match_couperet(a, b, elos_live, historique_live)
    lignes_bracket, champion, troisieme = simuler_tableau(seiziemes, sim)
    bracket_live = pd.DataFrame(lignes_bracket)

    # Sauvegarde dans des fichiers SEPARES (on ne touche pas aux fichiers geles)
    pred_live.to_csv("data/predictions_groupes_live.csv", index=False)
    qualifies_df(pred_live).to_csv("data/qualifies_1_16_live.csv", index=False)
    bracket_live.to_csv("data/bracket_complet_live.csv", index=False)
    pd.DataFrame([{"role": "Champion", "equipe": champion},
                  {"role": "3e place", "equipe": troisieme}]).to_csv("data/vainqueur_final_live.csv", index=False)

    print(f"CHAMPION LIVE : {champion} (3e : {troisieme})")
    print("Ecrits : predictions_groupes_live.csv, qualifies_1_16_live.csv, bracket_complet_live.csv, vainqueur_final_live.csv")


if __name__ == "__main__":
    main()