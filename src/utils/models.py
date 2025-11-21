import xgboost as xgb

def load_model(model_path):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model
