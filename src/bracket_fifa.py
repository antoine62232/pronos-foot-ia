"""Tableau a elimination directe de la CdM 2026 : construction du bracket et
simulation tour par tour."""

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


def _tour(n):
    return ("1/16 de finale" if n < 89 else "1/8 de finale" if n < 97 else
            "Quarts de finale" if n < 101 else "Demi-finales" if n < 104 else "Finale")


def simuler_tableau(seiziemes, simuler):
    """seiziemes : {num: (eq1, eq2)}. simuler : (eq1, eq2) -> (gagnant, confiance).
    Renvoie (lignes, champion, troisieme)."""
    vainqueur, lignes = {}, []
    matchs = {**dict(seiziemes), **{n: None for n in ARBRE}}
    for n in sorted(seiziemes) + sorted(ARBRE):
        if matchs[n] is None:
            a, b = ARBRE[n]
            matchs[n] = (vainqueur[a], vainqueur[b])
        e1, e2 = matchs[n]
        g, conf = simuler(e1, e2)
        vainqueur[n] = g
        lignes.append({"match": n, "tour": _tour(n), "equipe_1": e1, "equipe_2": e2,
                       "vainqueur": g, "confiance": round(conf, 1)})

    perdants = [matchs[m][1] if vainqueur[m] == matchs[m][0] else matchs[m][0] for m in PETITE_FINALE]
    troisieme, conf3 = simuler(perdants[0], perdants[1])
    lignes.append({"match": 103, "tour": "Petite finale", "equipe_1": perdants[0],
                   "equipe_2": perdants[1], "vainqueur": troisieme, "confiance": round(conf3, 1)})
    return lignes, vainqueur[104], troisieme