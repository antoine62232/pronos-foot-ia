import base64
import time
from pathlib import Path
import streamlit as st

def afficher_intro():
    """Affiche une intro video plein ecran, une seule fois par session."""
    
    # 1. Initialisation du chronomètre réel lors de la première arrivée de l'utilisateur
    if "intro_start_time" not in st.session_state:
        st.session_state["intro_start_time"] = time.time()
    
    temps_ecoule = time.time() - st.session_state["intro_start_time"]
    
    # Passé 8 secondes (marge de sécurité après les 5s d'animation), on arrête d'injecter 
    # ce code pour alléger définitivement le DOM lors des interactions futures.
    if temps_ecoule > 8.0:
        return

    chemin = Path("assets/intro.mp4")
    if not chemin.exists():
        return  

    video_b64 = base64.b64encode(chemin.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        /* --- 1. SÉCURISATION DE L'INTERFACE PRINCIPALE (ANTI-FLASH) --- */
        /* On force tous les blocs structurels de l'accueil à suivre l'animation de révélation.
           Le délai négatif (ex: -1.5s) indique au navigateur d'appliquer instantanément 
           l'état exact de l'animation à cet instant T, sans jamais repasser par la case départ. */
        .element-container:not(:has(#intro-overlay)), 
        div[data-testid="stHorizontalBlock"], 
        .stTabs,
        [data-testid="stMetricCard"] {{
            animation: reveal-ui 5s forwards !important;
            animation-delay: -{temps_ecoule}s !important;
        }}

        /* --- 2. ANCRAGE DE L'INTRO SUR L'ÉCRAN PHYSIQUE --- */
        #rideau-noir-absolu {{
            position: fixed;
            top: 0; left: 0; 
            width: 100vw; height: 100vh;
            background-color: #0A0E1A;
            z-index: 999998; 
            animation: intro-fade 5s forwards !important;
            animation-delay: -{temps_ecoule}s !important;
        }}

        #intro-overlay {{
            position: fixed;
            top: 0; left: 0; 
            width: 100vw; height: 100vh;
            z-index: 999999;          
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
        
        /* --- 3. TIMELINES DES ANIMATIONS SYNCHRONISÉES --- */
        /* L'intro reste opaque à 100% puis s'estompe proprement entre la 4e et 5e seconde (80% -> 100%) */
        @keyframes intro-fade {{
            0%   {{ opacity: 1; visibility: visible; }}
            80%  {{ opacity: 1; visibility: visible; }}
            100% {{ opacity: 0; visibility: hidden; pointer-events: none; }}
        }}
        
        /* L'interface d'accueil reste invisible (opacity: 0) et n'apparaît en douceur qu'à la toute fin */
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