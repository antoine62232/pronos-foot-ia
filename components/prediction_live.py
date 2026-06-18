# ====================================================================
# ROLE : Encart comparant la prediction d'AVANT-tournoi (gelee) a la
#        prediction LIVE, recalculee avec les vrais resultats deja joues.
#        Les deux champions viennent de fichiers SEPARES : on ne melange
#        jamais la prediction figee (qui sert a l'onglet Realite VS IA) et
#        la version live.
# ====================================================================

import streamlit as st
import pandas as pd
from pathlib import Path
from components.data_loader import get_url_drapeau

FICHIER_GELE = Path("data/vainqueur_final.csv")
FICHIER_LIVE = Path("data/vainqueur_final_live.csv")


def _champion(fichier):
    """Renvoie le champion lu dans un vainqueur_final*.csv, ou None si absent/illisible."""
    if not fichier.exists():
        return None
    try:
        df = pd.read_csv(fichier)
        ligne = df[df["role"] == "Champion"]
        return ligne.iloc[0]["equipe"] if len(ligne) else None
    except Exception:
        return None


def _bloc_equipe(label, equipe, couleur_label, couleur_nom):
    """Un bloc vertical : libelle + drapeau + nom, centre."""
    return (
        f'<div style="text-align:center;">'
        f'  <div style="color:{couleur_label}; font-size:12px; text-transform:uppercase;'
        f'              letter-spacing:1px; margin-bottom:8px;">{label}</div>'
        f'  <img src="{get_url_drapeau(equipe, 80)}" style="width:54px; border-radius:4px; margin-bottom:6px;">'
        f'  <div style="color:{couleur_nom}; font-size:18px; font-weight:700;">{equipe}</div>'
        f'</div>'
    )


def afficher_encart_champion_live():
    """Encart 'Avant le tournoi -> Aujourd'hui' pour le champion predit.

    Si la prediction live n'existe pas encore (robot pas encore passe), on
    n'affiche rien : l'onglet garde son comportement normal.
    """
    gele = _champion(FICHIER_GELE)
    live = _champion(FICHIER_LIVE)

    if gele is None or live is None:
        return

    if gele == live:
        # L'IA n'a pas change d'avis : message sobre plutot qu'une fausse opposition.
        contenu = (
            f'<div style="text-align:center;">'
            f'  <div style="color:#94A3B8; font-size:12px; text-transform:uppercase;'
            f'              letter-spacing:1px; margin-bottom:8px;">Favori confirmé par les premiers matchs</div>'
            f'  <img src="{get_url_drapeau(live, 80)}" style="width:54px; border-radius:4px; margin-bottom:6px;">'
            f'  <div style="color:#F8FAFC; font-size:18px; font-weight:700;">{live}</div>'
            f'</div>'
        )
    else:
        contenu = (
            f'<div style="display:flex; align-items:center; justify-content:center; gap:28px;">'
            f'  {_bloc_equipe("Avant le tournoi", gele, "#94A3B8", "#CBD5E1")}'
            f'  <div style="color:#A78BFA; font-size:28px;">&#8594;</div>'
            f'  {_bloc_equipe("Aujourd\'hui", live, "#A78BFA", "#F8FAFC")}'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1E1B4B,#141B2D); border:1px solid #1E293B;'
        f' border-radius:16px; padding:22px; margin-bottom:24px;">'
        f'  <div style="text-align:center; color:#64748B; font-size:12px; margin-bottom:16px;">'
        f'    🔄 Prédiction du champion, réévaluée avec les résultats déjà joués'
        f'  </div>'
        f'  {contenu}'
        f'</div>',
        unsafe_allow_html=True
    )