import json
import requests
import pytest
import pandas as pd
from api import app

@pytest.fixture
def client():
    return app.test_client()

def test_predict(client):
    
    # Créez des données de client au format JSON
    client = [{
                "SK_ID_CURR": 100875,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "AMT_INCOME_TOTAL": 20002,
                "AMT_CREDIT": 20000,
                "AMT_ANNUITY": 2000,
            }]

    client_json = json.dumps(client)

    # Envoyez une requête POST à la route /predict
    URL = "https://scoring-credit.streamlit.app/predict"
    res = requests.post(URL, json=client_json)
    
    print("client_json : ", client_json)
    print("res : ", res)
    
    data = res #.json()
    print("data : ", data)

    # pred = data["prediction"]
    # proba = data["probability"]

    ######################################################
    # Mes tests :
    ######################################################
    
    # Vérifiez le code de réponse HTTP
    assert res.status_code == 200

    # Vérification que data n'est pas vide
    assert data is not None
    
    # # Vérifiez la présence de l'élément "prediction" dans les données
    # assert "prediction" in data
    
    # # Vérifiez la présence de l'élément "probability" dans les données
    # assert "probability" in data
    
    # # Vérifiez que "pred" est un entier
    # assert isinstance(pred, int)
    
    # # Vérifiez que "proba" est un nombre réel (float)
    # assert isinstance(proba, float)
    
    # # Vérifiez que "pred" est soit 0 ou 1
    # assert pred in [0, 1]
    
    # # Vérifiez que "proba" est dans la plage de 0 à 1
    # assert 0 <= proba <= 1
