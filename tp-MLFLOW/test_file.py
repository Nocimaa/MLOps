import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"

# Données : features + résultat attendu (charges)
TEST_DATA = [
    {
        "input": {
            "age": 47, "sex": "female", "bmi": 24.32,
            "children": 0, "smoker": "no", "region": "northeast"
        },
        "expected": 10853.6718
    },
    {
        "input": {
            "age": 20, "sex": "male", "bmi": 39.4,
            "children": 2, "smoker": "yes", "region": "southwest"
        },
        "expected": 38344.566
    },
    {
        "input": {
            "age": 62, "sex": "male", "bmi": 30.875,
            "children": 3, "smoker": "yes", "region": "northwest"
        },
        "expected": 46718.16325
    },
]

@pytest.mark.parametrize("case", TEST_DATA)
def test_prediction(case):
    """Teste que l'API retourne un résultat proche du vrai charges."""
    url = f"{BASE_URL}/predict"  # adapte le endpoint selon ton API
    response = requests.post(url, json=case["input"])
    
    # Vérifie que l'API répond
    assert response.status_code == 200
    
    # Récupère la prédiction
    result = response.json()
    
    # On suppose que l'API renvoie {"prediction": valeur}
    pred = result.get("prediction")
    assert pred is not None, f"Réponse invalide: {result}"
    
    # Vérifie que la prédiction est proche de la valeur attendue (tolérance 15%)
    expected = case["expected"]
    assert abs(pred - expected) / expected < 0.10, f"Prédiction trop éloignée: {pred} vs {expected}"
