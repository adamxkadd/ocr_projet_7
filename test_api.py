import json
import pytest
from api import app

# Créez un client de test Flask 
@pytest.fixture
def client():
    app.config["TESTING"] = True
    client = app.test_client()
    yield client

# Testez la route /predict avec des données factices
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
    response = client.post("https://scoring-credit.streamlit.app/predict", data=client_data_json, content_type="application/json")

    # Vérifiez le code de réponse HTTP
    assert response.status_code == 200

    # Analysez la réponse JSON
    data = json.loads(response.data)

    # Vérifiez les valeurs renvoyées
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
