from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import xgboost as xgb
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Carga el modelo de XGBoost
# Cargar el modelo desde el archivo pkl
with open('docs/modelo.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Definir la ruta principal
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ruta para la predicción
@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()
    features = [[float(form_data['feature1'])], [float(form_data['feature2'])], [float(form_data['feature3'])], [float(form_data['feature4'])], [float(form_data['feature5'])]]  # Agrega todas las características necesarias para la predicción
    print(features)
    # Realizar la predicción con el modelo cargado
    data = pd.DataFrame(np.array(features)).T.set_axis(['dayofweek', 'quarter', 'month', 'year', 'dayofyear'], axis=1)
    prediction = modelo.predict(data)
    
    return templates.TemplateResponse("prediction.html", {"request": request, "prediction": prediction[0], "dayofweek": int(form_data['feature1']), "quarter": int(form_data['feature2']), "month":int(form_data['feature3']), "year":int(form_data['feature4']), "dayofyear":int(form_data['feature5'])})

