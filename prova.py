if __name__ == "__main__":
    import dvc.api
    import joblib

    with dvc.api.open('artifacts/models/model.joblib', repo='.') as f:
        model = joblib.load(f)

    print(type(model))
