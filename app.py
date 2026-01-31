import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Credit Card Prediction",
    page_icon="💳",
    layout="centered"
)

# Fonction pour charger le CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Charger le CSS
local_css("style.css")

# Chargement du modèle optimisé
@st.cache_resource
def load_model():
    return joblib.load('model_svm.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Fichier 'model_svm.pkl' introuvable. Veuillez exécuter le notebook d'entraînement d'abord ou uploader le fichier .pkl.")
    st.stop()

# --- HEADER ---
st.title("💳 Prédiction des Dépenses")
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    Cette application utilise un modèle <b>Support Vector Machine (SVR)</b> optimisé 
    pour estimer les dépenses mensuelles d'un client.
</div>
""", unsafe_allow_html=True)

# --- FORMULAIRE ---
st.write("### 📝 Informations du Client")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Variables numériques
        income = st.number_input("Revenu Annuel (en 10k $)", 0.5, 20.0, 4.5, step=0.1)
        age = st.number_input("Âge", 18, 100, 35)
        months = st.number_input("Mois à l'adresse actuelle", 0, 500, 24)
        share = st.number_input("Part du revenu (Ratio)", 0.0, 1.0, 0.05, format="%.4f")
        reports = st.number_input("Rapports négatifs (Défauts)", 0, 20, 0)

    with col2:
        # Variables numériques suite
        dependents = st.number_input("Personnes à charge", 0, 10, 1)
        majorcards = st.number_input("Nombre de cartes majeures", 0, 5, 1)
        active = st.number_input("Comptes actifs", 0, 30, 5)
        
        # Variables Catégorielles (Selectbox)
        card_input = st.selectbox("Demande de carte acceptée ?", ["Oui", "Non"])
        owner_input = st.selectbox("Propriétaire ?", ["Oui", "Non"])
        selfemp_input = st.selectbox("Indépendant ?", ["Oui", "Non"])

    # Conversion des inputs textuels en numérique (0 ou 1)
    card = 1 if card_input == "Oui" else 0
    owner = 1 if owner_input == "Oui" else 0
    selfemp = 1 if selfemp_input == "Oui" else 0

    submitted = st.form_submit_button("Lancer la Prédiction 🚀")

# --- RÉSULTAT ---
if submitted:
    # Création du tableau de données dans le bon ordre
    # Ordre strict : card, reports, age, income, share, owner, selfemp, dependents, months, majorcards, active
    features = np.array([[card, reports, age, income, share, owner, selfemp, dependents, months, majorcards, active]])
    
    # Prédiction (Le pipeline gère le scaling automatiquement !)
    prediction = model.predict(features)[0]
    
    st.markdown("---")
    
    # Affichage stylisé
    st.subheader("Résultat de l'analyse")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.metric(label="Dépenses Estimées", value=f"{prediction:.2f} $")
    
    with c2:
        if prediction < 50:
             st.info("Ce client a un profil de **faibles dépenses**.")
        elif prediction < 500:
             st.success("Ce client a un profil de dépenses **modérées**.")
        else:
             st.warning("Ce client a un profil de dépenses **élevées**.")