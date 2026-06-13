import base64
import time
from pathlib import Path
import streamlit as st
from streamlit_local_storage import LocalStorage


def afficher_intro():
    """Intro video plein ecran, jouee une fois et desactivable via le bouton Passer."""

    localS = LocalStorage()

    # Deja passee pendant cette session
    if st.session_state.get("intro_passee"):
        return

    # Deja passee lors d'une visite precedente (memorise dans le navigateur)
    if localS.getItem("intro_passee"):
        return

    if "intro_start_time" not in st.session_state:
        st.session_state["intro_start_time"] = time.time()

    temps_ecoule = time.time() - st.session_state["intro_start_time"]

    # L'animation dure 5s, au-dela plus besoin d'injecter le code
    if temps_ecoule > 8.0:
        return

    chemin = Path("assets/intro.mp4")
    if not chemin.exists():
        return

    video_b64 = base64.b64encode(chemin.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        /* L'accueil reste masque pendant l'intro, sauf le bouton Passer */
        .element-container:not(:has(#intro-overlay)):not(.st-key-passer_intro),
        div[data-testid="stHorizontalBlock"],
        .stTabs,
        [data-testid="stMetricCard"] {{
            animation: reveal-ui 5s forwards !important;
            animation-delay: -{temps_ecoule}s !important;
        }}

        #rideau-noir-absolu {{
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-color: #0A0E1A;
            z-index: 999998;
            pointer-events: none;
            animation: intro-fade 5s forwards !important;
            animation-delay: -{temps_ecoule}s !important;
        }}

        #intro-overlay {{
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            z-index: 999999;
            pointer-events: none;
            background: #0A0E1A;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: intro-fade 5s forwards !important;
            animation-delay: -{temps_ecoule}s !important;
        }}

        #intro-overlay video {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}

        /* Bouton Passer par-dessus l'overlay, seul element cliquable de l'intro */
        .st-key-passer_intro {{
            position: fixed !important;
            top: 22px;
            right: 32px;
            z-index: 1000000 !important;
            pointer-events: auto !important;
            animation: intro-fade 5s forwards !important;
            animation-delay: -{temps_ecoule}s !important;
        }}
        .st-key-passer_intro button {{
            background: rgba(255, 255, 255, 0.10);
            color: #E2E8F0;
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 999px;
            padding: 4px 18px;
            pointer-events: auto !important;
        }}

        @keyframes intro-fade {{
            0%   {{ opacity: 1; visibility: visible; }}
            80%  {{ opacity: 1; visibility: visible; }}
            100% {{ opacity: 0; visibility: hidden; pointer-events: none; }}
        }}

        @keyframes reveal-ui {{
            0%   {{ opacity: 0; pointer-events: none; }}
            90%  {{ opacity: 0; pointer-events: none; }}
            100% {{ opacity: 1; pointer-events: auto; }}
        }}
        </style>

        <div id="rideau-noir-absolu"></div>
        <div id="intro-overlay">
            <video autoplay muted playsinline>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # On memorise le choix pour la session et pour les prochaines visites.
    # Pas de st.rerun ici : il couperait la sauvegarde avant qu'elle parte dans le navigateur.
    if st.button("Passer l'intro", key="passer_intro"):
        st.session_state["intro_passee"] = True
        localS.setItem("intro_passee", "oui")