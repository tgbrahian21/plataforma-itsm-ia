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
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    palabras = texto.split()
    return " ".join([lemmatizer.lemmatize(p) for p in palabras if p not in stop_words])

def entrenar_xgboost_mejorado(ruta_datos):
    print("1. Cargando 15,000 datos (Triplicando el aprendizaje)...")
    df = pd.read_csv(ruta_datos) 
    
    df = df.sample(n=15000, random_state=42)
    df = df.dropna(subset=['issue_description', 'category', 'sla_breached'])
    
    conteo = df['category'].value_counts()
    df = df[df['category'].isin(conteo[conteo >= 10].index)] # Filtro más estricto
    
    print("2. Limpiando texto...")
    df['descripcion_limpia'] = df['issue_description'].apply(limpiar_texto)
    
    encoder_cat = LabelEncoder()
    y_cat = encoder_cat.fit_transform(df['category'])
    encoder_sla = LabelEncoder()
    y_sla = encoder_sla.fit_transform(df['sla_breached'])
    
    
    vectorizador = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), max_df=0.9, min_df=5)
    X_vect = vectorizador.fit_transform(df['descripcion_limpia'])
    
    
    print("\n3. Entrenando XGBoost Categorías (Ajuste Avanzado)...")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_vect, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
    
    
    modelo_cat = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    modelo_cat.fit(X_train_c, y_train_c)
    
    print("--- Métricas Mejoradas Categoría ---")
    print(classification_report(y_test_c, modelo_cat.predict(X_test_c), target_names=encoder_cat.classes_, zero_division=0))
    
    
    print("\n4. Entrenando XGBoost SLA con SMOTE...")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_vect, y_sla, test_size=0.2, random_state=42, stratify=y_sla)
    
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train_s, y_train_s)
    modelo_sla = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    modelo_sla.fit(X_smote, y_smote)
    
    print("--- Métricas Mejoradas SLA ---")
    print(classification_report(y_test_s, modelo_sla.predict(X_test_s), target_names=encoder_sla.classes_, zero_division=0))
    
    print("\n5. Sobrescribiendo modelos XGBoost con la versión PRO...")
    joblib.dump(vectorizador, 'src/models/tfidf_vectorizer_xgb.pkl')
    joblib.dump(encoder_cat, 'src/models/encoder_categoria.pkl')
    joblib.dump(encoder_sla, 'src/models/encoder_sla.pkl')
    joblib.dump(modelo_cat, 'src/models/modelo_categoria_xgboost.pkl')
    joblib.dump(modelo_sla, 'src/models/modelo_sla_xgboost.pkl')

if __name__ == "__main__":
    entrenar_xgboost_mejorado("data/customer_support_tickets_200k.csv")