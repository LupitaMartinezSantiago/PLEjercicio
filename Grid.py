from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

app = Flask(__name__)


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@app.route('/')
def home():
    return render_template('index.html', message="Sube tus datos para entrenar el modelo.")

@app.route('/train', methods=['POST'])
def train_model():
    
    param_grid = [
        {'n_estimators': [100, 500, 1000], 'max_leaf_nodes': [16, 24, 36]},
        {'bootstrap': [False], 'n_estimators': [100, 500]},
    ]

   
    rnd_clf = RandomForestClassifier(n_jobs=-1, random_state=42)

   
    grid_search = GridSearchCV(rnd_clf, param_grid, cv=5,
                               scoring='f1_weighted', return_train_score=True)

    # Entrenar
    grid_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y resultados
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Evaluar el modelo
    y_pred = grid_search.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    response = {
        "best_params": best_params,
        "best_score": best_score,
        "classification_report": report
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
