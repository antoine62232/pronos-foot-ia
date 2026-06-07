# ====================================================================
# ROLE : Compteur de visites (service gratuit Abacus, sans inscription)
# ====================================================================

import requests
import streamlit as st

# Ton compteur = un namespace + une cle (a toi). Garde-les uniques.
_URL = "https://abacus.jasoncameron.dev/hit/pronos-foot-ia-antoine-C20D25/visites"


def _incrementer():
    """Incremente le compteur de 1 et renvoie le nouveau total (None si souci)."""
    try:
        reponse = requests.get(_URL, timeout=5)
        return reponse.json().get("value")
    except Exception:
        return None   # service indisponible -> on n'affiche rien, pas de plantage


def afficher_compteur_visites():
    """Compte la visite (1 seule fois par session) et affiche le total."""
    # On ne compte qu'une fois par session : sinon chaque clic ferait +1
    if "visite_comptee" not in st.session_state:
        st.session_state["visite_comptee"] = True
        st.session_state["nb_visites"] = _incrementer()

    valeur = st.session_state.get("nb_visites")
    if valeur is None:
        return   # service indispo : on n'affiche simplement pas le compteur

    total = f"{valeur:,}".replace(",", " ")   # 1234 -> "1 234"
    st.markdown(
        f'<div style="text-align:center; margin: 0 0 18px;">'
        f'<span style="display:inline-flex; align-items:center; gap:6px; background:#141B2D;'
        f' border:1px solid #1E293B; color:#94A3B8; font-size:13px; padding:4px 14px; border-radius:20px;">'
        f'👀 <strong style="color:#F8FAFC;">{total}</strong> visites</span>'
        f'</div>',
        unsafe_allow_html=True
    )