import requests

api_url = "https://fastapi-app-dvesia.onrender.com"

data = {
    "age": 19,
    "workclass": "Private",
    "education": "HS-grad",
    "maritalStatus": "Never-married",
    "occupation": "Other-service",
    "relationship": "Own-child",
    "race": "Black",
    "sex": "Male",
    "hoursPerWeek": 40,
    "nativeCountry": "United-States"
}

response = requests.post(api_url, json=data)  # Add json=data to send the data with the request
print(response)
print("Response content:", response.content)
