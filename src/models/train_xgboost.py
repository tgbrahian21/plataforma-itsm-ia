import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    palabras = texto.split()
    palabras_limpias = [lemmatizer.lemmatize(p) for p in palabras if p not in stop_words]
    return " ".join(palabras_limpias)

def entrenar_xgboost(ruta_datos):
    print("1. Cargando datos (Misma semilla para comparación)...")
    df = pd.read_csv(ruta_datos) 
    df = df.sample(n=5000, random_state=42)
    df = df.dropna(subset=['issue_description', 'category', 'sla_breached'])
    
    # Filtro de seguridad: Quitar categorías con menos de 5 tickets
    conteo_categorias = df['category'].value_counts()
    categorias_validas = conteo_categorias[conteo_categorias >= 5].index
    df = df[df['category'].isin(categorias_validas)]
    
    print("2. Limpiando texto...")
    df['descripcion_limpia'] = df['issue_description'].apply(limpiar_texto)
    X = df['descripcion_limpia']
    
    # XGBoost requiere etiquetas numéricas (0, 1, 2...)
    encoder_cat = LabelEncoder()
    y_categoria_num = encoder_cat.fit_transform(df['category'])
    
    encoder_sla = LabelEncoder()
    y_sla_num = encoder_sla.fit_transform(df['sla_breached'])
    
    vectorizador = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_vect = vectorizador.fit_transform(X)
    
    # --- MODELO CATEGORÍAS ---
    print("3. Entrenando XGBoost para Categorías...")
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X_vect, y_categoria_num, test_size=0.2, random_state=42, stratify=y_categoria_num
    )
    modelo_cat = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    modelo_cat.fit(X_train_cat, y_train_cat)
    
    print("--- Métricas Categoría (XGBoost) ---")
    y_pred_cat = modelo_cat.predict(X_test_cat)
    print(classification_report(y_test_cat, y_pred_cat, target_names=encoder_cat.classes_, zero_division=0))
    
    # --- MODELO SLA (CON SMOTE) ---
    print("\n4. Entrenando XGBoost para SLA con SMOTE...")
    X_train_sla, X_test_sla, y_train_sla, y_test_sla = train_test_split(
        X_vect, y_sla_num, test_size=0.2, random_state=42, stratify=y_sla_num
    )
    
    smote = SMOTE(random_state=42)
    X_train_sla_smote, y_train_sla_smote = smote.fit_resample(X_train_sla, y_train_sla)
    
    modelo_sla = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    modelo_sla.fit(X_train_sla_smote, y_train_sla_smote)
    
    print("--- Métricas Riesgo SLA (XGBoost + SMOTE) ---")
    y_pred_sla = modelo_sla.predict(X_test_sla)
    print(classification_report(y_test_sla, y_pred_sla, target_names=encoder_sla.classes_, zero_division=0))
    
    print("\n5. Guardando todos los modelos...")
    joblib.dump(vectorizador, 'src/models/tfidf_vectorizer_xgb.pkl')
    joblib.dump(encoder_cat, 'src/models/encoder_categoria.pkl')
    joblib.dump(encoder_sla, 'src/models/encoder_sla.pkl')
    joblib.dump(modelo_cat, 'src/models/modelo_categoria_xgboost.pkl')
    joblib.dump(modelo_sla, 'src/models/modelo_sla_xgboost.pkl')
    print("¡Finalizado!")

if __name__ == "__main__":
    entrenar_xgboost("data/customer_support_tickets_200k.csv")