# ====================================================================
# ROLE : Confronter les pronostics de l'IA aux resultats reels,
#        lus dans data/resultats_reels.csv (rempli par le robot GitHub Actions)
# ====================================================================

import streamlit as st
import pandas as pd
from pathlib import Path

from components.dates import _date_fr

FICHIER_RESULTATS = Path("data/resultats_reels.csv")


def _charger_resultats():
    """Lit les resultats reels et renvoie un index {paire d'equipes -> score}.
    Tant qu'aucun match n'est joue, le fichier est vide -> on renvoie {}."""
    if not FICHIER_RESULTATS.exists():
        return {}
    try:
        df = pd.read_csv(FICHIER_RESULTATS)
    except pd.errors.EmptyDataError:
        return {}

    index = {}
    for _, m in df.iterrows():
        cle = frozenset({m["equipe_domicile"], m["equipe_exterieur"]})
        index[cle] = {
            "home": m["equipe_domicile"],
            "sd": int(m["buts_domicile"]),
            "se": int(m["buts_exterieur"]),
        }
    return index


# Correspondance : TON nom d'equipe  ->  nom utilise dans le CSV (football-data.org).
# A completer le 11 juin si un nom differe (ex: "South Korea": "Korea Republic").
# Si une equipe n'est pas ici, on suppose que les deux noms sont identiques.
CORRESPONDANCE_NOMS = {
    # "South Korea": "Korea Republic",
}


def _nom_reel(equipe):
    """Renvoie le nom tel qu'il apparait dans le CSV (via le dico), sinon inchange."""
    return CORRESPONDANCE_NOMS.get(equipe, equipe)


def _issue(score_dom, score_ext):
    """Traduit un score en resultat : '1' (dom), 'N' (nul) ou '2' (ext)."""
    if score_dom > score_ext:
        return "1"
    if score_dom == score_ext:
        return "N"
    return "2"


def comparer_predictions(df_predictions):
    """
    Compare chaque prediction de poule au resultat reel (si le match est joue).
    Renvoie (lignes, nb_corrects, nb_joues).
    - lignes : liste de dicts {dom, ext, prono, statut, score, correct, date}
    - statut : "joue" ou "attente"
    """
    index_reel = _charger_resultats()

    lignes, nb_corrects, nb_joues = [], 0, 0
    for _, p in df_predictions.iterrows():
        dom_n, ext_n = _nom_reel(p["equipe_dom"]), _nom_reel(p["equipe_ext"])
        cle = frozenset({dom_n, ext_n})

        # Pas (encore) dans le CSV -> match a venir
        if cle not in index_reel:
            lignes.append({"dom": p["equipe_dom"], "ext": p["equipe_ext"],
                           "prono": p["pronostic"], "statut": "attente",
                           "score": None, "correct": None, "date": p["date"]})
            continue

        # On oriente le score selon TON domicile/exterieur (le CSV peut les inverser)
        r = index_reel[cle]
        sd, se = (r["sd"], r["se"]) if r["home"] == dom_n else (r["se"], r["sd"])
        correct = (_issue(sd, se) == p["pronostic"])

        nb_joues += 1
        nb_corrects += int(correct)
        lignes.append({"dom": p["equipe_dom"], "ext": p["equipe_ext"],
                       "prono": p["pronostic"], "statut": "joue",
                       "score": (sd, se), "correct": correct, "date": p["date"]})

    return lignes, nb_corrects, nb_joues

# ====================================================================
# L'ONGLET VISIBLE
# ====================================================================

def _texte_prono(prono, dom, ext):
    """Traduit le pronostic en texte lisible."""
    if prono == "1":
        return f"Victoire {dom}"
    if prono == "2":
        return f"Victoire {ext}"
    return "Match nul"


def afficher_onglet_resultats(df_predictions):
    """Onglet : confronte les pronostics de l'IA aux vrais resultats."""
    st.subheader("Réalité VS IA")
    st.markdown(
        '<p style="color: #94A3B8;">Le duel : les pronostics de l\'IA face aux vrais résultats, '
        'mis à jour automatiquement pendant la compétition.</p>',
        unsafe_allow_html=True
    )

    lignes, nb_corrects, nb_joues = comparer_predictions(df_predictions)

    # --- Le score global de l'IA ---
    if nb_joues == 0:
        st.info(
            "⏳ Aucun match de Coupe du Monde n'a encore été joué. "
            "Le score de l'IA s'affichera ici **automatiquement** dès le coup d'envoi (11 juin 2026) !"
        )
    else:
        pct = round(nb_corrects / nb_joues * 100)
        st.markdown(
            f'<div style="text-align: center; padding: 24px; margin-bottom: 24px;'
            f' background: linear-gradient(135deg, #1E1B4B, #141B2D);'
            f' border: 1px solid #1E293B; border-radius: 16px;">'
            f'  <div style="color: #94A3B8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Score de l\'IA</div>'
            f'  <div style="color: #A78BFA; font-size: 48px; font-weight: 800; margin: 6px 0;">{pct}%</div>'
            f'  <div style="color: #F8FAFC; font-size: 15px;">{nb_corrects} bons pronostics sur {nb_joues} matchs joués</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    joues = [l for l in lignes if l["statut"] == "joue"]
    attente = [l for l in lignes if l["statut"] == "attente"]

    # --- Les matchs joues (prediction vs realite) ---
    if joues:
        st.markdown("### Matchs joués")
        for l in joues:
            sd, se = l["score"]
            ok = l["correct"]
            couleur = "#10B981" if ok else "#F87171"
            badge = (f'<span style="color:{couleur}; font-weight:bold;">✅ Réussi</span>' if ok
                     else f'<span style="color:{couleur}; font-weight:bold;">❌ Manqué</span>')
            st.markdown(
                f'<div style="background:#141B2D; border:1px solid #1E293B; border-left:4px solid {couleur};'
                f' padding:14px; border-radius:8px; margin-bottom:10px;'
                f' display:flex; justify-content:space-between; align-items:center;">'
                f'  <div>'
                f'    <div style="color:#F8FAFC; font-weight:600;">{l["dom"]} {sd} - {se} {l["ext"]}</div>'
                f'    <div style="color:#94A3B8; font-size:12px; margin-top:2px;">{_date_fr(l["date"])} · Pronostic IA : {_texte_prono(l["prono"], l["dom"], l["ext"])}</div>'
                f'  </div>'
                f'  <div style="font-size:13px;">{badge}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # --- Les matchs a venir ---
    if attente:
        with st.expander(f"⏳ Matchs à venir ({len(attente)})"):
            for l in attente:
                st.markdown(
                    f'<div style="padding:6px 0; border-bottom:1px solid #1E293B;">'
                    f'<span style="color:#64748B; font-size:12px;">{_date_fr(l["date"])} · </span>'
                    f'<span style="color:#CBD5E1;">{l["dom"]} vs {l["ext"]}</span>'
                    f'<span style="color:#64748B; font-size:12px;"> — Pronostic IA : {_texte_prono(l["prono"], l["dom"], l["ext"])}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )