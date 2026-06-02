import base64
from pathlib import Path
import streamlit as st


def afficher_intro():
    """Affiche une intro video plein ecran, une seule fois par session."""
    # Si l'intro a deja ete vue dans cette session, on ne fait rien.
    if st.session_state.get("intro_vue"):
        return
    st.session_state["intro_vue"] = True

    chemin = Path("assets/intro.mp4")
    if not chemin.exists():
        return  # securite : si le fichier manque, l'app continue normalement

    # On encode la video en base64 pour l'embarquer directement dans la page.
    video_b64 = base64.b64encode(chemin.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        #intro-overlay {{
            position: fixed;
            inset: 0;                 /* couvre tout l'ecran */
            z-index: 999999;          /* au-dessus de toute l'app */
            background: #0A0E1A;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: intro-fade 5s forwards;   /* visible ~4s puis fondu 1s */
        }}
        #intro-overlay video {{
            width: 100%;
            height: 100%;
            object-fit: contain;      /* affiche toute la video (passe a 'cover' pour remplir) */
        }}
        @keyframes intro-fade {{
            0%   {{ opacity: 1; }}
            80%  {{ opacity: 1; }}
            100% {{ opacity: 0; visibility: hidden; pointer-events: none; }}
        }}
        </style>

        <div id="intro-overlay">
            <video autoplay muted playsinline>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )