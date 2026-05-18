import streamlit as st
import pandas as pd
from components.match_card import afficher_carte_match, formater_pronostic


def afficher_onglet_mes_pronos(df_predictions):
    """Version TEST diagnostic - on isole le probleme."""

    # ========== TEST 1 : Texte simple ==========
    st.write("TEST 1 - Si tu vois ce texte, la fonction est appelee correctement")

    # ========== TEST 2 : Subheader ==========
    st.subheader("TEST 2 - Defie l'IA")

    # ========== TEST 3 : Verifier df_predictions ==========
    st.write(f"TEST 3 - Nombre de matchs : {len(df_predictions)}")

    # ========== TEST 4 : st.info ==========
    st.info("TEST 4 - Si tu vois cette boite bleue, st.info marche")

    # ========== TEST 5 : st.session_state ==========
    if "test_compteur" not in st.session_state:
        st.session_state.test_compteur = 0

    st.write(f"TEST 5 - Compteur session_state : {st.session_state.test_compteur}")

    # ========== TEST 6 : Selectbox ==========
    options = ["Option A", "Option B", "Option C"]
    choix = st.selectbox("TEST 6 - Selectbox :", options)
    st.write(f"Tu as choisi : {choix}")

    # ========== TEST 7 : Import de match_card ==========
    try:
        # On essaie d'utiliser une fonction importee
        premier_match = df_predictions.iloc[0]
        st.write(f"TEST 7 - Premier match : {premier_match['equipe_dom']} vs {premier_match['equipe_ext']}")
    except Exception as e:
        st.error(f"TEST 7 ERREUR : {e}")

    # ========== TEST 8 : afficher_carte_match ==========
    try:
        st.write("TEST 8 - Affichage de la carte du premier match :")
        afficher_carte_match(df_predictions.iloc[0])
    except Exception as e:
        st.error(f"TEST 8 ERREUR : {e}")