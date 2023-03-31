import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """
    Create a test client for the app.
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    """
    Test the GET method.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Greetings!"}


def test_get_malformed(client):
    """
    Test GET with a malformed URL.
    """
    response = client.get("/wrong_url")
    assert response.status_code != 200


def test_post_malformed(client):
    """
    Test POST with malformed input data.
    """
    data = {
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "ERROR",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    }
    response = client.post("/", json=data)
    assert response.status_code == 422
