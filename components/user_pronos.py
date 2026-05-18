import streamlit as st
import pandas as pd
from components.match_card import afficher_carte_match, formater_pronostic


def afficher_onglet_mes_pronos(df_predictions):
    """
    Affiche l'onglet interactif "Mes pronos".

    L'utilisateur peut saisir ses propres pronostics et les comparer
    avec ceux de l'IA. Tous les pronos sont memorises grace a
    st.session_state pour ne pas etre perdus.

    Parametre :
        df_predictions (DataFrame) : Les 72 predictions de l'IA
    """

    # INITIALISATION DE LA MEMOIRE (st.session_state)

    # On verifie si la cle "mes_pronos" existe deja dans la session
    # Si non, on l'initialise avec un dictionnaire vide
    # Ce dictionnaire survivra aux re-executions de Streamlit
    if "mes_pronos" not in st.session_state:
        st.session_state.mes_pronos = {}

    # HEADER

    st.subheader("Defie l'Intelligence Artificielle")
    st.markdown(
        '<p style="color: #94A3B8;">Selectionne un match et compare ton pronostic a celui de la machine. Tes pronos sont memorises pendant ta session.</p>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # COMPTEUR DE PRONOS

    # On affiche en haut combien de pronos l'utilisateur a deja faits
    nb_pronos_faits = len(st.session_state.mes_pronos)
    nb_pronos_total = len(df_predictions)

    # Calcul du nombre de pronos en accord avec l'IA
    nb_accord_ia = sum(
        1 for prono in st.session_state.mes_pronos.values()
        if prono["accord_avec_ia"]
    )

    # Affichage en 3 colonnes pour avoir un look "dashboard"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pronos enregistres", f"{nb_pronos_faits} / {nb_pronos_total}")
    with col2:
        st.metric("Accord avec l'IA", f"{nb_accord_ia} / {nb_pronos_faits}" if nb_pronos_faits > 0 else "0 / 0")
    with col3:
        if nb_pronos_faits > 0:
            pourcentage = (nb_accord_ia / nb_pronos_faits) * 100
            st.metric("Taux d'accord", f"{pourcentage:.0f}%")
        else:
            st.metric("Taux d'accord", "-")

    st.markdown("<br>", unsafe_allow_html=True)

    # SELECTION DU MATCH

    liste_matchs_texte = [
        f"{row['equipe_dom']} vs {row['equipe_ext']} ({row['date']})"
        for _, row in df_predictions.iterrows()
    ]

    match_selectionne_txt = st.selectbox(
        "Choisis le match a pronostiquer :",
        liste_matchs_texte
    )

    # On retrouve la ligne correspondante
    index_match = liste_matchs_texte.index(match_selectionne_txt)
    match_choisi = df_predictions.iloc[index_match]

    eq_dom = match_choisi['equipe_dom']
    eq_ext = match_choisi['equipe_ext']

    # Cle unique pour identifier le match dans le dictionnaire
    cle_match = f"{eq_dom}_{eq_ext}_{match_choisi['date']}"

    # AFFICHAGE DE LA CARTE DU MATCH

    afficher_carte_match(match_choisi)

    # INDICATEUR SI DEJA PRONOSTIQUE

    # Si l'utilisateur a deja fait un prono pour ce match, on l'indique
    if cle_match in st.session_state.mes_pronos:
        ancien_prono = st.session_state.mes_pronos[cle_match]
        st.info(f"Tu as deja pronostique ce match : **{ancien_prono['mon_choix']}**. Tu peux modifier ton choix ci-dessous.")

    # FORMULAIRE DE PRONOSTIC

    with st.form("formulaire_pronostics_user"):
        st.markdown("### Ton pronostic")

        choix_user = st.radio(
            "Quel resultat predis-tu ?",
            [f"Victoire {eq_dom}", "Match Nul", f"Victoire {eq_ext}"],
            horizontal=True
        )

        bouton_valider = st.form_submit_button("Enregistrer mon prono")

        # TRAITEMENT APRES VALIDATION

        if bouton_valider:
            prono_ia_txt = formater_pronostic(match_choisi['pronostic'], eq_dom, eq_ext)

            # Mapping pour comparaison simple
            mapping_choix_to_code = {
                f"Victoire {eq_dom}": '1',
                "Match Nul": 'N',
                f"Victoire {eq_ext}": '2',
            }
            code_choix_user = mapping_choix_to_code[choix_user]
            accord_avec_ia = (code_choix_user == match_choisi['pronostic'])

            # SAUVEGARDE DANS st.session_state
            # On stocke toutes les infos utiles pour le recap
            st.session_state.mes_pronos[cle_match] = {
                "match": match_selectionne_txt,
                "equipe_dom": eq_dom,
                "equipe_ext": eq_ext,
                "date": str(match_choisi['date']),
                "mon_choix": choix_user,
                "choix_ia": prono_ia_txt,
                "accord_avec_ia": accord_avec_ia,
            }

            st.success("Pronostic enregistre !")

            # Recap visuel apres validation
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Ton choix**")
                st.info(choix_user)
            with col2:
                st.markdown("**Choix de l'IA**")
                st.info(prono_ia_txt)

            # Probabilites IA
            st.markdown("**Selon l'IA**")
            col_p1, col_pN, col_p2 = st.columns(3)
            with col_p1:
                st.metric(eq_dom, f"{match_choisi['proba_1']:.0f}%")
            with col_pN:
                st.metric("Match nul", f"{match_choisi['proba_N']:.0f}%")
            with col_p2:
                st.metric(eq_ext, f"{match_choisi['proba_2']:.0f}%")

            if accord_avec_ia:
                st.balloons()
                st.success("Tu penses exactement comme l'IA. Statistiquement, tu maximises tes chances !")
            else:
                st.warning("Tu entres en dualite avec la machine. Que le meilleur gagne.")

    # RECAPITULATIF DE TOUS LES PRONOS

    if nb_pronos_faits > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Recapitulatif de mes pronos")

        # CONSTRUCTION DU DATAFRAME RECAP

        liste_pour_tableau = []
        for cle, prono in st.session_state.mes_pronos.items():
            liste_pour_tableau.append({
                "Date": prono["date"],
                "Match": f"{prono['equipe_dom']} vs {prono['equipe_ext']}",
                "Mon prono": prono["mon_choix"],
                "Prono IA": prono["choix_ia"],
                "Accord": "Oui" if prono["accord_avec_ia"] else "Non",
            })

        df_recap = pd.DataFrame(liste_pour_tableau)

        # FILTRE D'AFFICHAGE

        # st.radio horizontal pour choisir le filtre
        filtre = st.radio(
            "Filtrer :",
            ["Tous les pronos", "Seulement les desaccords avec l'IA", "Seulement les accords avec l'IA"],
            horizontal=True,
            label_visibility="collapsed"  # Cache le label "Filtrer :"
        )

        # On applique le filtre choisi sur le DataFrame
        if filtre == "Seulement les desaccords avec l'IA":
            df_affiche = df_recap[df_recap["Accord"] == "Non"]
        elif filtre == "Seulement les accords avec l'IA":
            df_affiche = df_recap[df_recap["Accord"] == "Oui"]
        else:
            df_affiche = df_recap

        # AFFICHAGE DU TABLEAU

        # Si le filtre ne renvoie aucun resultat, on affiche un message
        if len(df_affiche) == 0:
            st.info("Aucun pronostic ne correspond a ce filtre.")
        else:
            st.dataframe(df_affiche, use_container_width=True, hide_index=True)

        # ACTIONS : EXPORT CSV + RESET

        st.markdown("<br>", unsafe_allow_html=True)
        col_export, col_reset = st.columns(2)

        with col_export:
            # BOUTON D'EXPORT CSV
            # df.to_csv() transforme le DataFrame en texte CSV
            # On le stocke dans la variable csv_data
            csv_data = df_recap.to_csv(index=False).encode('utf-8')

            # st.download_button cree un vrai bouton de telechargement
            # Quand l'utilisateur clique, son navigateur telecharge le fichier
            st.download_button(
                label="Telecharger mes pronos (CSV)",
                data=csv_data,
                file_name="mes_pronos_coupe_du_monde_2026.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_reset:
            # BOUTON DE REINITIALISATION
            # Quand on clique, on vide le dictionnaire dans session_state
            # st.rerun() force Streamlit a recharger la page pour reflet
            if st.button("Reinitialiser tous mes pronos", use_container_width=True):
                st.session_state.mes_pronos = {}
                st.rerun()