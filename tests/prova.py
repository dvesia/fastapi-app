from joblib import load

if __name__ == "__main__":
    print(type(load("../artifacts/models/lb.joblib")))