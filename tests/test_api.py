from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_predict_species():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    # Fetch the API token from environment or hardcode for testing
    api_token = os.getenv("API_TOKEN", "75SSDZH4WE177T8K")
    response = client.post(
        "/predict",
        json=payload,
        headers={"X-API-Token": api_token}
    )
    assert response.status_code == 200
    assert "predicted_species" in response.json()
    # Optionally: assert response.json()["predicted_species"] == "setosa"
