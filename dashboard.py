import os
import pickle
import json
import requests
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import hydralit_components as hc
import subprocess
import sys
import shap

shap.initjs()

URL = "http://127.0.0.1:5000/predict" 

@st.cache_data(persist=True)
def run_api():
    subprocess.Popen([sys.executable, 'api.py'])

run_api()

@st.cache_data(persist=True)
def deserialization():
    file = open("features_exp.pkl", 'rb')
    explainer, features, feature_names = pickle.load(file)
    file.close()
    return explainer, features, feature_names

explainer, features, feature_names = deserialization()

@st.cache_data(persist=True)
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(path="data.csv")

@st.cache_data(persist=True)
def split_data(df, num_rows):
    X = df.iloc[:, 2:]
    y = df["TARGET"]
    ids = df["SK_ID_CURR"]
    
    _, X_test, _, y_test, _, ids = train_test_split(X, y, ids, test_size=0.2, random_state=42, stratify=y)
    
    X_test = X_test.iloc[:num_rows, ]
    y_test = y_test.iloc[:num_rows, ]
    ids = list(ids[:num_rows, ])
    return X_test, y_test, ids

X_test, y_test, ids = split_data(df=df, num_rows=1000)

@st.cache_data(persist=True)
def model_prediction(input):
    req = requests.post(URL, json=input, timeout=120).json()
    return req["prediction"], req["probability"]

def main():
    # st.set_page_config(layout="wide") 
    st.title('SCORING CREDIT BANCAIRE')
    st.title(" ")
	
    menu_data = [
				{'label':"Left End"},
				{'label':"Book"},
				{'label':"Component"},
				{'label':"Dashboard"},
				{'label':"Right End"},
				]
    menu_id = hc.option_bar(horizontal_orientation=True)
    menu_id = hc.nav_bar(menu_definition=menu_data)
    st.info(f"{menu_id=}")

    menu_data = [
            {'icon': "far fa-address-book", 'label': "Prédiction"}, 
            {'icon': "far fa-chart-bar", 'label': "Importance des Caractéristiques"},
            {'icon': "fas fa-tachometer-alt", 'label': "Analyse des Données"},
            {'icon': "fas fa-folder-plus", 'label': "Nouveau Client"},
            {'icon': "far fa-list-alt", 'label': "Classement des clients"}
    ]
    
    over_theme = {'txc_inactive': '#FFFFFF', 'menu_background': '#20B2AA'}
    page = hc.option_bar(
							option_definition=menu_data,
							key='PrimaryOption',
							override_theme=over_theme,
							horizontal_orientation=True
						)

    st.sidebar.header("DASHBORD")
    
    upload_file = st.sidebar.file_uploader("Télécharger les données", type=["csv"])
    if upload_file:
        df_up = pd.read_csv(upload_file)
        df_analysis = df_up.copy()

    df_analysis = df.copy()
    for col in df_analysis.filter(like="DAYS").columns:
        df_analysis[col] = df_analysis[col].apply(lambda x: abs(x / 365))
    df_analysis.columns = df_analysis.columns.str.replace("DAYS", "YEARS")
    df_analysis["TARGET"] = df_analysis["TARGET"].astype(str)
    choice_list = list(df_analysis.iloc[:, 2:].columns)

    if page == "Analyse des Données":
        st.title("Exploration des Données")
        data_analysis = st.sidebar.radio(
            "Choisir le type d'analyse:",
            ["Univariée", "Multivariée"],
            index=0,
        )

        if data_analysis == "Univariée":
            st.header("Analyse Univariée")
            options = st.multiselect(
                "Choisir la variable à analyser",
                choice_list,
                ["AMT_INCOME_TOTAL", "AMT_CREDIT", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "YEARS_BIRTH"]
            )

            if df_analysis[options].select_dtypes(include=["int64", "float64"]).shape[1] > 0:
                graphic_style = st.sidebar.radio(
                    "Sélectionner le type de graphique",
                    ("Histogramme", "Boîte à Moustaches"),
                    index=0,
                )

            if len(options) > 1:
                col1, col2 = st.columns(2)

            for i in range(len(options)):
                if df_analysis[options[i]].dtype == "object":
                    data = df_analysis.groupby("TARGET")[options[i]].value_counts().reset_index(name="pourcentage")
                    data["pourcentage"] = (data["pourcentage"] / len(df_analysis) * 100).round(1)
                    fig = px.bar(
                        data,
                        x=options[i],
                        y="pourcentage",
                        color="TARGET",
                        color_discrete_sequence=px.colors.qualitative.Pastel2,
                    )
                    if len(options) > 1:
                        if i % 2 == 0:
                            col1.plotly_chart(fig, use_container_width=True)
                        else:
                            col2.plotly_chart(fig, use_container_width=True)
                    else:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    if graphic_style == "Box Plot":
                        fig = px.box(
                            df_analysis,
                            labels=options[i],
                            y=options[i],
                            points='suspectedoutliers',
                            color="TARGET",
                            category_orders={"TARGET": ["0", "1"]},
                            color_discrete_sequence=px.colors.qualitative.Pastel2,
                        )                       
                    else:
                        fig = px.histogram(
                            df_analysis,
                            x=options[i],
                            color="TARGET",
                            category_orders={"TARGET": ["0", "1"]},
                            histnorm="percent",
                            nbins=10,
                            color_discrete_sequence=px.colors.qualitative.Pastel2,
                        )
                        fig.update_layout(bargap=0.1)
                    if len(options) > 1:
                        if i % 2 == 0:
                            col1.plotly_chart(fig, use_container_width=True)
                        else:
                            col2.plotly_chart(fig, use_container_width=True)
                    else:
                        st.plotly_chart(fig, use_container_width=True)

        else:
            num_choice_list = list(df_analysis.iloc[:, 2:].select_dtypes(include=["int64", "float64"]).columns)

            container = st.container()
            options = container.multiselect(
                "Choisir plusieurs variables numériques à analyser:",
                num_choice_list,
                ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_2", "EXT_SOURCE_3"],
            )
            if len(options) > 0:
                import plotly.io as pio
                pio.templates.default = "none"
                corr = df_analysis[["TARGET"] + options]
                corr["TARGET"] = corr["TARGET"].astype(int)
                corr = corr.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                fig, ax = plt.subplots()
                sns.heatmap(corr, ax=ax,annot=True, fmt=".2f", mask=mask, center=0, cmap="coolwarm")
                plt.title(f"Heatmap des corrélations linéaires\n")
                st.write(fig)
            else:
                st.warning("Veuillez sélectionner au moins 1 variable")

    elif page == "Prédiction":
        sorted_ids = sorted(ids)
        client_id = st.selectbox("Sélectionner l'ID du client:", sorted_ids,)
        
        id_idx = ids.index(client_id)
        client_input = X_test.iloc[[id_idx], :]

        st.header("Effectuer la prédiction pour le client : {}".format(client_id))
        
        with st.expander("Afficher les informations sur le client :"):
            df_client_input = pd.DataFrame( client_input.to_numpy(), index=["Information"],columns=client_input.columns,).astype(str) #.transpose()
            st.dataframe(df_client_input)

        if st.button("Prédire"):
            client_input_json = json.loads(client_input.to_json())
            pred, proba = model_prediction(client_input_json)
            
            if pred == 0:
                st.write('<div style="color:green;text-align:center;font-size:50px;font-weight:bold;">Prêt Accordé</div>', unsafe_allow_html=True)
                st.success("Probabilité de défaut : {}%".format(proba)) 
            else:
                st.write('<div style="color:red;text-align:center;font-size:50px;font-weight:bold;">Prêt Refusé</div>', unsafe_allow_html=True)
                st.error("Probabilité de défaut : {}%".format(proba))
            
            st.expander("Afficher l'impact des caractéristiques:")
            force_plot, ax = plt.subplots()
            force_plot = shap.force_plot(
                                            base_value=explainer.expected_value[pred],
                                            shap_values=explainer.shap_values[pred][id_idx],
                                            features=features[id_idx],
                                            plot_cmap=["#00e800", "#ff2839"],
                                            feature_names=feature_names,
                                            matplotlib=True,
                                            show=False,
                                        )
            st.write(force_plot)

            decision_plot, ax = plt.subplots()
            ax = shap.decision_plot(
                                        base_value=explainer.expected_value[pred],
                                        shap_values=explainer.shap_values[pred][id_idx],
                                        features=features[id_idx],
                                        feature_names=feature_names,
                                        link='logit',
                                    )
            st.pyplot(decision_plot)

    elif page == "Importance des Caractéristiques":
        st.title("Importance des Caractéristiques pour la Prédiction")
        n_features = st.slider(
                                "Sélectionner le nombre de caractéristiques:",
                                value=7,
                                min_value=5,
                                max_value=50,
                                step=2
                              )
        summary_plot, _ = plt.subplots(2, 1)
        
        plt.subplot(121)
        shap.summary_plot(
                            shap_values=explainer.shap_values[1],
                            features=features,
                            feature_names=feature_names,
                            max_display=n_features,
                            plot_size=[6, 2 + (n_features/5)],
                            color_bar=False
                        )
        plt.title("Crédit raté", size=20)
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])

        plt.subplot(122)
        shap.summary_plot(
                            shap_values=explainer.shap_values[0],
                            features=features,
                            feature_names=feature_names,
                            max_display=n_features,
                            plot_size=[6, 2 + (n_features/5)]
                        )
        plt.yticks([])
        plt.xticks([])
        plt.title("Crédit réussi", size=20)
        plt.xlabel("")
        st.pyplot(summary_plot)

    elif page == "Nouveau Client":
        client_median = X_test.iloc[[1],:]
        st.title('Simulation')
        my_form = st.form(key='form-1')
        client_median['AMT_CREDIT'] = my_form.text_input('Crédit du prêt :', "20000")
        client_median['AMT_GOODS_PRICE'] = my_form.text_input('Crédit précédent :', "2200200")
        client_median['AMT_ANNUITY'] = my_form.text_input('Annuité AMT :', "2000")
        client_median['AMT_INCOME_TOTAL'] = my_form.text_input('Revenu total :', "20002")
        my_form.radio('Genre', ('M', 'F'))
        my_form.slider('Âge :', 18, 120, 25)

        if my_form.form_submit_button('Soumettre'):
            client_input_json = json.loads(client_median.to_json())
            pred, proba = model_prediction(client_input_json)
            if pred == 0:
                st.success("Prêt accordé (probabilité de remboursement = {}%)".format(proba)) 
            else:
                st.error("Prêt refusé (probabilité de défaut = {}%)".format(proba))
        
        with st.expander("Afficher les informations sur le client :"):
            df_client_input = pd.DataFrame(
                client_median.to_numpy(),
                index=["Information"],
                columns=client_median.columns,
            ).astype(str).transpose()
            st.dataframe(df_client_input)

if __name__ == "__main__":
    main()
