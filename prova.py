if __name__ == "__main__":
    import dvc.api
    import joblib

    with dvc.api.open('/Users/dvesia/PycharmProjects/fastapi-deployment/artifacts/models/model.joblib', repo='.') as f:
        print(f)
