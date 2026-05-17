import streamlit as st
from components.match_card import _formater_pronostic 

def afficher_onglet_mes_pronos(df_predictions):
    """
    Affiche le formulaire interactif permettant à l'utilisateur
    de saisir ses pronostics et de les comparer à l'IA.
    
    Paramètre :
        df_predictions (DataFrame) : Le tableau contenant les 72 prédictions de l'IA
    """
    st.subheader("Défie l'Intelligence Artificielle")
    st.markdown(
        '<p style="color: #94A3B8;">Sélectionne un match de la compétition et compare ton pronostic à celui de la machine.</p>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Création d'une liste textuelle propre pour la liste déroulante
    # Exemple de rendu : "France 🆚 Spain"
    liste_matchs_texte = [
        f"{row['equipe_dom']} 🆚 {row['equipe_ext']}" 
        for _, row in df_predictions.iterrows()
    ]

    # 2. Affichage de la boîte de sélection (Selectbox)
    match_selectionne_txt = st.selectbox(
        "Choisis le match à pronostiquer :",
        liste_matchs_texte
    )

    # 3. On retrouve la ligne correspondante dans notre DataFrame grâce à l'index sélectionné
    index_match = liste_matchs_texte.index(match_selectionne_txt)
    match_choisi = df_predictions.iloc[index_match]

    # Extraction des variables utiles pour ce match spécifique
    eq_dom = match_choisi['equipe_dom']
    eq_ext = match_choisi['equipe_ext']

    # 4. Bloc formulaire pour isoler les clics de l'utilisateur
    with st.form("formulaire_pronostics_user"):
        st.write(f"**Match sélectionné :** {eq_dom} contre {eq_ext}")
        
        # Choix de l'utilisateur via des boutons radio
        choix_user = st.radio(
            "Quel est ton prono ?",
            [f"Victoire {eq_dom}", "Match Nul", f"Victoire {eq_ext}"]
        )
        
        # Bouton de soumission obligatoire dans un st.form
        bouton_valider = st.form_submit_button("Enregistrer mon prono")

        # Logique de traitement après validation
        if bouton_valider:
            st.success("Ton pronostic a bien été pris en compte pour ce match !")
            
            # On transforme le prono brut de l'IA ('1', 'N', '2') en texte lisible
            prono_ia_txt = _formater_pronostic(match_choisi['pronostic'], eq_dom, eq_ext)
            
            st.write(f"Ton choix : **{choix_user}**")
            st.write(f"Le choix de l'IA : **{prono_ia_txt}**")

            # 5. Vérification des divergences ou accords
            # On vérifie si la chaîne de caractères choisie par l'utilisateur correspond au résultat de l'IA
            if (match_choisi['pronostic'] == '1' and f"Victoire {eq_dom}" == choix_user) or \
               (match_choisi['pronostic'] == '2' and f"Victoire {eq_ext}" == choix_user) or \
               (match_choisi['pronostic'] == 'N' and "Match Nul" == choix_user):
                st.balloons() # Animation festive !
                st.info("Tu penses exactement comme l'IA. Statisquement, vous maximisez vos chances !")
            else:
                st.warning("Tu entres en dualité avec la machine ! Que le meilleur gagne.")