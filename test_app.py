from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_get_items():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Greetings!"}


def test_inference():
    data = {
        "age": 30,
        "workclass": "Private",
        "education": "Bachelors",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States",
    }
    response = client.post("/", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
