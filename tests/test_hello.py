from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_hello_with_valid_name():
    response = client.post("/hello", json={"name": "John"})
    assert response.status_code == 200      # All OK
    assert response.json() == {"message": "Hello John"}

def test_hello_with_empty_name():
    response = client.post("/hello", json={"name": ""})
    assert response.status_code == 200
    assert response.json() == {"message": "Hello "}

def test_hello_missing_name_field():
    response = client.post("/hello", json={})
    assert response.status_code == 422

def test_hello_with_non_string_name():
    response = client.post("/hello", json={"name": 123})
    assert response.status_code == 422