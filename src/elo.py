"""Calcul d'un classement Elo pour les equipes nationales."""

from collections import defaultdict
import pandas as pd

# K : vitesse d'ajustement du classement. 40 = valeur de depart classique.
K = 40
# Bonus accorde a l'equipe qui recoit, sauf sur terrain neutre.
AVANTAGE_DOMICILE = 100


def proba_victoire(elo_dom, elo_ext, neutre=False):
    """Probabilite theorique que l'equipe a domicile gagne, selon l'ecart d'Elo."""
    avantage = 0 if neutre else AVANTAGE_DOMICILE
    return 1 / (1 + 10 ** (-(elo_dom + avantage - elo_ext) / 400))


def maj_elo(elo_dom, elo_ext, score_dom, score_ext, neutre=False):
    """Renvoie les nouveaux Elo (domicile, exterieur) apres un match."""
    attendu_dom = proba_victoire(elo_dom, elo_ext, neutre)
    if score_dom > score_ext:
        reel_dom = 1.0
    elif score_dom == score_ext:
        reel_dom = 0.5
    else:
        reel_dom = 0.0
    nouveau_dom = elo_dom + K * (reel_dom - attendu_dom)
    nouveau_ext = elo_ext + K * ((1 - reel_dom) - (1 - attendu_dom))
    return nouveau_dom, nouveau_ext


def calculer_elo(df):
    """Parcourt les matchs dans l'ordre des dates et ajoute l'Elo d'AVANT-match.

    Renvoie le df trie par date avec deux colonnes : elo_domicile, elo_exterieur.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Chaque equipe demarre a 1500 ; le dictionnaire memorise le classement courant.
    ratings = defaultdict(lambda: 1500.0)
    elo_dom_list, elo_ext_list = [], []

    for ligne in df.itertuples(index=False):
        elo_dom = ratings[ligne.home_team]
        elo_ext = ratings[ligne.away_team]

        # On enregistre l'Elo AVANT le match : c'est la feature, sans aucune fuite.
        elo_dom_list.append(elo_dom)
        elo_ext_list.append(elo_ext)

        # Puis on met a jour les classements pour les matchs suivants.
        ratings[ligne.home_team], ratings[ligne.away_team] = maj_elo(
            elo_dom, elo_ext, ligne.home_score, ligne.away_score, bool(ligne.neutral)
        )

    df["elo_domicile"] = elo_dom_list
    df["elo_exterieur"] = elo_ext_list
    return df


def classement_actuel(df, top=15):
    """Classement Elo le plus recent par equipe (utile pour verifier et pour l'app)."""
    dom = df[["date", "home_team", "elo_domicile"]].rename(columns={"home_team": "equipe", "elo_domicile": "elo"})
    ext = df[["date", "away_team", "elo_exterieur"]].rename(columns={"away_team": "equipe", "elo_exterieur": "elo"})
    tout = pd.concat([dom, ext])
    dernier = tout.sort_values("date").groupby("equipe").tail(1)
    return dernier.sort_values("elo", ascending=False).head(top)


if __name__ == "__main__":
    # Verification : on calcule l'Elo sur tout l'historique et on affiche le top.
    df = pd.read_csv("data/matchs_entrainement.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = calculer_elo(df)

    print("Top 15 des equipes par Elo (le plus recent) :")
    classement = classement_actuel(df, top=15)
    for rang, ligne in enumerate(classement.itertuples(index=False), start=1):
        print(f"  {rang:2}. {ligne.equipe:20} {round(ligne.elo)}")

def elo_final(df):
    """Renvoie un dict {equipe: Elo courant} apres TOUS les matchs de l'historique.

    Sert au moment de predire : c'est l'Elo de chaque equipe juste avant son
    prochain match (donc son niveau actuel).
    """
    df = df.sort_values("date")
    ratings = defaultdict(lambda: 1500.0)
    for ligne in df.itertuples(index=False):
        ratings[ligne.home_team], ratings[ligne.away_team] = maj_elo(
            ratings[ligne.home_team], ratings[ligne.away_team],
            ligne.home_score, ligne.away_score, bool(ligne.neutral)
        )
    return dict(ratings)