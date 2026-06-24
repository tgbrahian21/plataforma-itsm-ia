from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = FastAPI(title="ITSM AI Platform", version="0.4.0")


try:
    rf_cat = joblib.load('src/models/modelo_categoria_baseline.pkl')
    rf_sla = joblib.load('src/models/modelo_sla_baseline.pkl')
except: rf_cat, rf_sla = None, None

try:
    xgb_cat = joblib.load('src/models/modelo_categoria_xgboost.pkl')
    xgb_sla = joblib.load('src/models/modelo_sla_xgboost.pkl')
    xgb_vectorizer = joblib.load('src/models/tfidf_vectorizer_xgb.pkl')
    xgb_enc_cat = joblib.load('src/models/encoder_categoria.pkl')
    xgb_enc_sla = joblib.load('src/models/encoder_sla.pkl')
except: xgb_cat = None

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    texto = re.sub(r'[^a-z\s]', '', str(texto).lower())
    return " ".join([lemmatizer.lemmatize(p) for p in texto.split() if p not in stop_words])

class TicketIn(BaseModel):
    descripcion: str
    modelo_elegido: str = "baseline"
    
class TicketOut(BaseModel):
    categoria_predicha: str
    confianza_categoria: float
    top_categorias: dict  
    prediccion_incumplimiento_sla: str
    riesgo_sla_porcentaje: float
    modelo_usado: str

@app.post("/predict", response_model=TicketOut)
def predecir_ticket(ticket: TicketIn):
    texto = limpiar_texto(ticket.descripcion)
    mod = ticket.modelo_elegido.lower()

    if mod == "baseline":
        pred_cat = str(rf_cat.predict([texto])[0])
        proba_cat_array = rf_cat.predict_proba([texto])[0]
        conf_cat = float(proba_cat_array.max())
        
    
        clases_rf = rf_cat.classes_
        top_3_idx = np.argsort(proba_cat_array)[-3:][::-1]
        top_cats = {str(clases_rf[i]): float(proba_cat_array[i]*100) for i in top_3_idx}

        res_sla = rf_sla.predict([texto])[0]
        prob_sla = rf_sla.predict_proba([texto])[0]
        idx_yes = list(rf_sla.classes_).index("Yes") if "Yes" in rf_sla.classes_ else 1
        riesgo = float(prob_sla[idx_yes])

    elif mod == "xgboost":
        X_vec = xgb_vectorizer.transform([texto])
        
        res_cat_num = xgb_cat.predict(X_vec)[0]
        pred_cat = str(xgb_enc_cat.inverse_transform([res_cat_num])[0])
        proba_cat_array = xgb_cat.predict_proba(X_vec)[0]
        conf_cat = float(proba_cat_array.max())
        
        
        top_3_idx = np.argsort(proba_cat_array)[-3:][::-1]
        top_cats = {str(xgb_enc_cat.inverse_transform([i])[0]): float(proba_cat_array[i]*100) for i in top_3_idx}

        res_sla_num = xgb_sla.predict(X_vec)[0]
        prob_sla = xgb_sla.predict_proba(X_vec)[0]
        idx_yes = list(xgb_enc_sla.classes_).index("Yes") if "Yes" in xgb_enc_sla.classes_ else 1
        riesgo = float(prob_sla[idx_yes])

    return TicketOut(
        categoria_predicha=pred_cat,
        confianza_categoria=round(conf_cat, 2),
        top_categorias=top_cats,
        prediccion_incumplimiento_sla=str("Yes" if riesgo >= 0.5 else "No"),
        riesgo_sla_porcentaje=round(riesgo * 100, 2),
        modelo_usado=mod
    )