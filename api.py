import os
import git
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Flask app initialisation
app = Flask(__name__)


# charger le modele
file = open("model.pkl", 'rb')
transformer, classifier = pickle.load(file)
file.close()


@app.route("/predict", methods=["POST","GET"])
def predict():
    try:
        # récuperer le df client
        client_input = request.get_json()
        client_input = pd.DataFrame(client_input)
        
        # effectuer la prediction
        client_input = transformer.transform(client_input)
        pred = classifier.predict(client_input)[0]
        proba = classifier.predict_proba(client_input)[0][pred]

        # envoyer la réponse
        return jsonify(prediction=int(pred), probability=round(100 * proba, 1))

    except Exception as e:
        return jsonify(error=str(e)), 500  


if __name__ == '__main__':
    app.run(debug=True)
