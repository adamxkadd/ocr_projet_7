import json
import requests
import pytest
import pandas as pd
from api import app

@pytest.fixture
def client():
    return app.test_client()
    
def test_predict(client):
    # Créez des données de client en df 
    client = [
        {
            "SK_ID_CURR": 100875,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "AMT_INCOME_TOTAL": 20002,
            "AMT_CREDIT": 20000,
            "AMT_ANNUITY": 2000,
        }
    ]
    
    client_df = pd.DataFrame(client)
    client_json = json.loads(client_df.to_json())
    print("client_json : ", client_json)
    # Convertissez les données en JSON 
    # client_data_json = json.dumps(client_data)
      
    # Envoyez une requête POST à la route /predict
    URL = "https://scoring-credit.streamlit.app/predict"
    response = requests.post(URL, json=client_json, timeout=120)
    print("response : ", response)
    
    data = response #.json()
    pred, proba = data["prediction"], data["probability"]
    print("Probabilité du risque : {}%".format(proba))
    # response = requests.post(URL, json=client_data, timeout=120)
    # data = response.json()
    
    # Vérifiez le code de réponse HTTP
    assert response.status_code == 200

    
    # # Vérifiez les valeurs renvoyées
    # assert "prediction" in data
    # assert "probability" in data
    # assert isinstance(data["prediction"], int)
    # assert isinstance(data["probability"], float)
