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


#def test_post_above(client):
#    """
#    Test POST with input features above the threshold.
#    """
#    data = {
#        "age": 32,
#        "workclass": "Private",
#        "education": "Some-college",
#        "maritalStatus": "Married-civ-spouse",
#        "occupation": "Exec-managerial",
#        "relationship": "Husband",
#        "race": "White",
#        "sex": "Male",
#        "hoursPerWeek": 60,
#        "nativeCountry": "United-States"
#    }
#    response = client.post("/", json=data)
#    assert response.status_code == 200
#    assert response.json() == {"prediction": ">50K"}
#
#
#def test_post_below(client):
#    """
#    Test POST with input features below the threshold.
#    """
#    data = {
#        "age": 19,
#        "workclass": "Private",
#        "education": "HS-grad",
#        "maritalStatus": "Never-married",
#        "occupation": "Other-service",
#        "relationship": "Own-child",
#        "race": "Black",
#        "sex": "Male",
#        "hoursPerWeek": 40,
#        "nativeCountry": "United-States"
#    }
#    response = client.post("/", json=data)
#    assert response.status_code == 200
#    assert response.json() == {"prediction": "<=50K"}


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
