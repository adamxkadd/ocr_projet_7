import os
import git
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Init Flask app
app = Flask(__name__)

# Load model objects
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as handle:
    transformer, classifier = pickle.load(handle)

    
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

    
    
    
    


# @app.route("/predict", methods=["POST","GET"])
# def predict():
#     # Parse data as JSON
#     client_input = request.get_json()
#     client_input = pd.DataFrame(client_input)
#     # client_input = transformer.transform(client_input)
#     # pred = classifier.predict(client_input)[0]
#     # proba = classifier.predict_proba(client_input)[0][pred]
#     pred = 0
#     proba = 0.275      
#     return jsonify(prediction=int(pred), probability=round(100 * proba, 1))


if __name__ == '__main__':
    app.run(debug=True)
