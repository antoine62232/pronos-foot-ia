"""Tableau a elimination directe de la CdM 2026 : construction du bracket et
simulation tour par tour. Les matchs deja joues prennent le resultat reel (vrai
vainqueur, a.p. / t.a.b. inclus), les autres sont simules par le modele."""

import pandas as pd

# Slots : "1X" = 1er du groupe X, "2X" = 2e du groupe X, "3" = un des meilleurs 3es
SEIZIEMES = {
    73: ("2A", "2B"), 74: ("1E", "3"),  75: ("1F", "2C"), 76: ("1C", "2F"),
    77: ("1I", "3"),  78: ("2E", "2I"), 79: ("1A", "3"),  80: ("1L", "3"),
    81: ("1D", "3"),  82: ("1G", "3"),  83: ("2K", "2L"), 84: ("1H", "2J"),
    85: ("1B", "3"),  86: ("1J", "2H"), 87: ("1K", "3"),  88: ("2D", "2G"),
}

# Chaque match du tour suivant recoit les vainqueurs des deux matchs indiques
ARBRE = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),
}
PETITE_FINALE = (101, 102)  # les deux perdants des demies jouent la 3e place (M103)

# Slot d'un 3e -> groupe du 3e qualifie, d'apres la table d'attribution FIFA du
# round of 32 pour les groupes des 8 meilleurs 3es retenus.
ATTRIB_3ES = {74: "D", 77: "F", 79: "E", 80: "K", 81: "B", 82: "I", 85: "J", 87: "L"}

# Libelle affiche selon la colonne "prolongation" de resultats_reels.csv.
# A ajuster si l'API renvoie d'autres valeurs pour les phases finales.
DECISION = {
    "REGULAR": "",
    "EXTRA_TIME": "a.p.",
    "PENALTY_SHOOTOUT": "t.a.b.",
    "PENALTIES": "t.a.b.",
}


def construire_seiziemes(premiers, deuxiemes, troisiemes_par_groupe):
    """premiers / deuxiemes / troisiemes_par_groupe : dict groupe -> equipe.
    Renvoie {num_match: (equipe_1, equipe_2)}."""
    attendus = set(ATTRIB_3ES.values())
    if set(troisiemes_par_groupe) != attendus:
        raise ValueError(
            f"3es qualifies {sorted(troisiemes_par_groupe)} != attribution FIFA "
            f"{sorted(attendus)}. Phase de groupes incomplete ou resultats_reels.csv pas a jour."
        )
    def equipe(slot, num):
        if slot == "3":
            return troisiemes_par_groupe[ATTRIB_3ES[num]]
        table = premiers if slot[0] == "1" else deuxiemes
        return table[slot[1]]
    return {num: (equipe(a, num), equipe(b, num)) for num, (a, b) in SEIZIEMES.items()}


def charger_resultats_elim(lignes, normaliser=None):
    """Construit le lookup des matchs a elimination directe deja joues.
    lignes : iterable de dict (lecture de resultats_reels.csv).
    normaliser : nom equipe brut -> nom equipe du bracket (memes regles que pour les poules).
    Renvoie {frozenset({eq1, eq2}): {"vainqueur", "decision", "score"}}."""
    norm = normaliser or (lambda x: x)
    reels = {}
    for r in lignes:
        if r["phase"] == "GROUP_STAGE":
            continue
        # Match sans score (en cours, donnee manquante) : inexploitable, on ignore.
        if pd.isna(r["buts_domicile"]) or pd.isna(r["buts_exterieur"]):
            continue
        dom, ext = norm(r["equipe_domicile"]), norm(r["equipe_exterieur"])
        bd, be = int(r["buts_domicile"]), int(r["buts_exterieur"])
        # football-data laisse "winner" vide sur une seance de tirs au but : on tranche au score.
        v = str(r["vainqueur"]).strip()
        if v == "HOME_TEAM":
            gagnant = dom
        elif v == "AWAY_TEAM":
            gagnant = ext
        elif bd != be:
            gagnant = dom if bd > be else ext
        else:
            continue  # nul sans vainqueur designe : inexploitable en elimination directe
        reels[frozenset((dom, ext))] = {
            "vainqueur": gagnant,
            "decision": DECISION.get(r["prolongation"], ""),
            "buts": {dom: bd, ext: be},
        }
    return reels


def _tour(n):
    return ("1/16 de finale" if n < 89 else "1/8 de finale" if n < 97 else
            "Quarts de finale" if n < 101 else "Demi-finales" if n < 104 else "Finale")


def _resoudre(num, tour, e1, e2, simuler, reels):
    """Renvoie la ligne du match : resultat reel si dispo, sinon simulation."""
    reel = reels.get(frozenset((e1, e2)))
    if reel:
        b = reel["buts"]
        score = f"{b[e1]}-{b[e2]}" if e1 in b and e2 in b else ""
        return {"match": num, "tour": tour, "equipe_1": e1, "equipe_2": e2,
                "vainqueur": reel["vainqueur"], "confiance": 100.0, "reel": True,
                "score": score, "decision": reel["decision"]}
    g, conf = simuler(e1, e2)
    return {"match": num, "tour": tour, "equipe_1": e1, "equipe_2": e2,
            "vainqueur": g, "confiance": round(conf, 1), "reel": False,
            "score": "", "decision": ""}


def simuler_tableau(seiziemes, simuler, resultats_reels=None):
    """seiziemes : {num: (eq1, eq2)}. simuler : (eq1, eq2) -> (gagnant, confiance).
    resultats_reels : {frozenset({eq1, eq2}): {...}} des matchs deja joues (optionnel,
    voir charger_resultats_elim). Renvoie (lignes, champion, troisieme)."""
    reels = resultats_reels or {}
    vainqueur, lignes = {}, []
    matchs = {**dict(seiziemes), **{n: None for n in ARBRE}}
    for n in sorted(seiziemes) + sorted(ARBRE):
        if matchs[n] is None:
            a, b = ARBRE[n]
            matchs[n] = (vainqueur[a], vainqueur[b])
        e1, e2 = matchs[n]
        ligne = _resoudre(n, _tour(n), e1, e2, simuler, reels)
        vainqueur[n] = ligne["vainqueur"]
        lignes.append(ligne)

    perdants = [matchs[m][1] if vainqueur[m] == matchs[m][0] else matchs[m][0] for m in PETITE_FINALE]
    ligne3 = _resoudre(103, "Petite finale", perdants[0], perdants[1], simuler, reels)
    lignes.append(ligne3)
    return lignes, vainqueur[104], ligne3["vainqueur"]