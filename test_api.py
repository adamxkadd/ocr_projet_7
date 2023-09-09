import json
import requests
import pytest
from api import app

@pytest.fixture
def client():
    return app.test_client()
    
def test_predict(client):
    # Créez des données de client factices au format JSON
    client_data = [
        {
          "SK_ID_CURR": 100875,
          "NAME_CONTRACT_TYPE": "Cash loans",
          "AMT_INCOME_TOTAL": 20002,
          "AMT_CREDIT": 20000,
          "AMT_ANNUITY": 2000,
        }
    ]

    # Convertissez les données en JSON
    client_data_json = json.dumps(client_data)
    
    # Envoyez une requête POST à la route /predict
    URL = "https://scoring-credit.streamlit.app/predict"
    response = requests.post(URL, json=client_data_json, timeout=120)

    # Vérifiez le code de réponse HTTP
    assert response.status_code == 200

    data = response

    # data["prediction"], 
    # data["probability"],

    # Vérifiez les valeurs renvoyées
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
