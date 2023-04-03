if __name__ == '__main__':
    import dvc.api
    from joblib import load

    with dvc.api.open(
            'model.joblib',
            mode='rb',
            remote='mybucket') as f:
        model = load(f)
