"""Met a jour les resultats reels de la CdM 2026 dans data/resultats_reels.csv.

Source : API football-data.org (plan gratuit, competition "WC").
A lancer en local pour tester, et automatiquement par GitHub Actions pendant le tournoi.
"""

import os
import tomllib
from pathlib import Path

import requests
import pandas as pd

URL_MATCHS = "https://api.football-data.org/v4/competitions/WC/matches"
FICHIER_SORTIE = Path("data/resultats_reels.csv")


def cle_api():
    # En ligne (GitHub Actions) la cle arrive par variable d'env ; en local on relit secrets.toml.
    cle = os.environ.get("FOOTBALL_DATA_KEY")
    if cle:
        return cle
    secrets = Path(".streamlit/secrets.toml")
    if secrets.exists():
        with open(secrets, "rb") as f:
            data = tomllib.load(f)
        if "FOOTBALL_DATA_KEY" in data:
            return data["FOOTBALL_DATA_KEY"]
    raise SystemExit("Cle FOOTBALL_DATA_KEY introuvable (secrets.toml en local, secret GitHub en ligne).")


def recuperer_matchs(cle):
    r = requests.get(URL_MATCHS, headers={"X-Auth-Token": cle}, timeout=20)
    r.raise_for_status()
    return r.json().get("matches", [])


def lignes_terminees(matchs):
    lignes = []
    for m in matchs:
        # On ne garde que les matchs joues ; "winner" tranche aussi ceux decides aux tirs au but.
        if m["status"] != "FINISHED":
            continue
        score = m["score"]
        lignes.append({
            "id_match": m["id"],
            "date": m["utcDate"][:10],
            "phase": m.get("stage"),
            "equipe_domicile": m["homeTeam"]["name"],
            "equipe_exterieur": m["awayTeam"]["name"],
            "buts_domicile": score["fullTime"]["home"],
            "buts_exterieur": score["fullTime"]["away"],
            "vainqueur": score["winner"],       # HOME_TEAM / AWAY_TEAM / DRAW
            "prolongation": score["duration"],  # REGULAR / EXTRA_TIME / PENALTY_SHOOTOUT
        })
    return lignes


def main():
    matchs = recuperer_matchs(cle_api())
    resultats = lignes_terminees(matchs)
    print(f"{len(matchs)} matchs WC recus, {len(resultats)} termines")

    df = pd.DataFrame(resultats)
    if not df.empty:
        df = df.sort_values("date")

    FICHIER_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")
    print(f"Ecrit : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()