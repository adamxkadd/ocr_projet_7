import json
import requests
import pytest
import pandas as pd
from api import app

@pytest.fixture
def client():
    return app.test_client()

def test_predict(client):
    # Créez des données de client factices au format JSON
    client = [
        {
            "SK_ID_CURR": 100875,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "AMT_INCOME_TOTAL": 20002,
            "AMT_CREDIT": 20000,
            "AMT_ANNUITY": 2000,
        }
    ]

    client_json = json.dumps(client)

    # Envoyez une requête POST à la route /predict
    URL = "https://scoring-credit.streamlit.app/predict"
    res = requests.post(URL, json=client_json)
    
    # client_df = pd.DataFrame(client_data)
    # client_json = client_df.to_json(orient='records')
    # # client_json = json.loads(client_df.to_json(orient='records'))
    # print("client_json : ", client_json)
    # # Convertissez les données en JSON 
    # # client_data_json = json.dumps(client_data)

    
    print("affichage  :>>>>>>>")
    print("client_json : ", client_json)
    print(res)
    # print(res.get_data(as_text=True))
    data = res.json()
    print(data)

    pred = data["prediction"]
    proba = data["probability"]

    ######################################################
    # Mes tests :
    ######################################################
    
    # Vérifiez le code de réponse HTTP
    assert res.status_code == 200
    
    # Vérifiez la présence de l'élément "prediction" dans les données
    assert "prediction" in data
    
    # Vérifiez la présence de l'élément "probability" dans les données
    assert "probability" in data
    
    # Vérifiez que "pred" est un entier
    assert isinstance(pred, int)
    
    # Vérifiez que "proba" est un nombre réel (float)
    assert isinstance(proba, float)
    
    # Vérifiez que "pred" est soit 0 ou 1
    assert pred in [0, 1]
    
    # Vérifiez que "proba" est dans la plage de 0 à 1
    assert 0 <= proba <= 1





# def test_predict(client):
#     # Créez des données de client en df 
#     client = [
#         {
#             "SK_ID_CURR": 100875,
#             "NAME_CONTRACT_TYPE": "Cash loans",
#             "AMT_INCOME_TOTAL": 20002,
#             "AMT_CREDIT": 20000,
#             "AMT_ANNUITY": 2000,
#         }
#     ]
    
#     client_df = pd.DataFrame(client)
#     client_json = json.loads(client_df.to_json())
#     print("client_json : ", client_json)
#     # Convertissez les données en JSON 
#     # client_data_json = json.dumps(client_data)
      
#     # Envoyez une requête POST à la route /predict
#     URL = "https://scoring-credit.streamlit.app/predict"
#     response = requests.post(URL, json=client_json, timeout=120)
#     print("response : ", response.text)

#     data = json.loads(response_text)
#     pred, proba = data["prediction"], data["probability"]
#     print("Probabilité du risque : {}%".format(proba))

#     # data = response #.json()
#     # pred, proba = data["prediction"], data["probability"]
#     # print("Probabilité du risque : {}%".format(proba))
#     # response = requests.post(URL, json=client_data, timeout=120)
#     # data = response.json()
    
#     # Vérifiez le code de réponse HTTP
#     assert response.status_code == 200

    
#     # # Vérifiez les valeurs renvoyées
#     # assert pred in data
#     # assert proba in data
#     # assert isinstance(data["prediction"], int)
#     # assert isinstance(data["probability"], float)
