### Openclassrooms
# Projet7 "Implémentez un modèle de scoring" 
Master : Data Scientist - 
Author : KADDOURI Abdelghani

Applications SCORING CRÉDIT

Objectif du projet 
--------------------
L’entreprise "Prêt à dépenser" souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité 
qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. 
Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données 
variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs 
de transparence vis-à-vis des décisions d’octroi de crédit. 
Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client 
puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, 
mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 


Découpage des dossiers
----------------------
- Kaddouri_Abdelghani_1_dashboard_et_API_082023.ipynb : L’application de dashboard interactif répondant aux spécifications ci-dessus et l’API de prédiction du score, déployées chacunes sur le cloud.
- Kaddouri_Abdelghani_2_dossier_code_082023 : le dossier géré via un outil de versioning de code contenant :
- dashboard.py : le code générant le dashboard
- api.py : le code permettant de déployer le modèle sous forme d'API
- pretraitement_prediction.ipynb : le notebook de la modélisation (du prétraitement à la prédiction), intégrant via MLFlow le tracking d’expérimentations et le stockage centralisé des modèles
- data_drift_evidently.ipynb : code generant le tableau HTML d’analyse de data drift réalisé à partir d’evidently
- ReadMe_objectif-du-projet.txt : fichier introductif permettant de comprendre l'objectif du projet et le découpage des dossiers
- requirements.txt : Fichier listant les packages utilisés 


Liens
-----
- Application dashboard : https://scoring-credit.streamlit.app/
- GitHub : https://github.com/adamxkadd/ocr_projet_7.git
- Données Kaggle : https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script 
- Vidéo démo :

