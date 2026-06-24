import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
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

def entrenar_modelos(ruta_datos):
    print("1. Cargando datos...")
    df = pd.read_csv(ruta_datos) 
    df = df.sample(n=5000, random_state=42)
    
    
    df = df.dropna(subset=['issue_description', 'category', 'sla_breached'])
    
    print("2. Limpiando texto...")
    df['descripcion_limpia'] = df['issue_description'].apply(limpiar_texto)
    
    
    X = df['descripcion_limpia']
    y_categoria = df['category'] 
    y_sla = df['sla_breached']
    
    X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_categoria, test_size=0.2, random_state=42)
    _, _, y_train_sla, y_test_sla = train_test_split(X, y_sla, test_size=0.2, random_state=42)
    
    print("3. Entrenando Modelo de Categorías...")
    pipeline_cat = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    pipeline_cat.fit(X_train, y_train_cat)
    joblib.dump(pipeline_cat, 'src/models/modelo_categoria_baseline.pkl')
    
    print("4. Entrenando Modelo de SLA...")
    pipeline_sla = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    pipeline_sla.fit(X_train, y_train_sla)
    joblib.dump(pipeline_sla, 'src/models/modelo_sla_baseline.pkl')
    
    print("¡Ambos modelos entrenados y guardados exitosamente!")

if __name__ == "__main__":
    entrenar_modelos("data/customer_support_tickets_200k.csv")