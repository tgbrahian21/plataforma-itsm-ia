# Plataforma ITSM Asistida por Inteligencia Artificial

Este repositorio contiene el Producto Mínimo Viable, MVP, de una plataforma ITSM, *IT Service Management*, desarrollada como Trabajo de Fin de Máster. El sistema utiliza técnicas de Procesamiento de Lenguaje Natural, NLP, y algoritmos de Machine Learning supervisado para automatizar el triaje de incidencias técnicas reportadas por los usuarios.

La solución tiene como propósito apoyar la gestión de tickets de soporte mediante modelos predictivos capaces de clasificar automáticamente las incidencias y estimar el riesgo de incumplimiento de los Acuerdos de Nivel de Servicio, SLA.

---

## 1. Objetivos Analíticos del Proyecto

La plataforma cumple dos objetivos analíticos principales:

1. **Clasificación Multiclase:**
   Inferir la categoría técnica del ticket a partir del reporte en lenguaje natural realizado por el usuario.

2. **Predicción Binaria de SLA:**
   Calcular de manera probabilística el riesgo de incumplimiento de los Acuerdos de Nivel de Servicio, SLA.

---

## 2. Arquitectura General

La arquitectura del sistema está dividida en dos componentes principales:

* **Backend asíncrono:** desarrollado con FastAPI.
* **Frontend interactivo:** desarrollado con Streamlit.

El sistema opera sobre dos motores predictivos comparativos:

* **Modelo Baseline:** Random Forest.
* **Modelo Retador:** XGBoost optimizado con SMOTE.

Esta estructura permite comparar el desempeño de un modelo base frente a un modelo más avanzado, con el fin de evaluar mejoras en precisión, balanceo de clases y capacidad predictiva.

---

## 3. Conjunto de Datos

Por motivos de optimización del control de versiones y debido a las políticas de GitHub respecto a archivos superiores a 100 MB, el conjunto de datos no está incluido directamente en este repositorio.

Para entrenar los modelos y ejecutar correctamente la plataforma, es necesario descargar el corpus documental oficial desde Kaggle.

**Fuente del Dataset:**
[Customer Support Tickets Dataset - 200K+ Records - Kaggle](https://www.kaggle.com/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records)

### 3.1. Instrucciones de integración de datos

1. Ingresar al enlace del dataset en Kaggle.
2. Descargar el archivo comprimido.
3. Extraer el archivo CSV original.
4. Renombrar el archivo a:

```bash
customer_support_tickets_200k.csv
```

5. Ubicar el archivo CSV directamente dentro de la carpeta `data/`, en la raíz del proyecto.

La ruta final esperada debe ser:

```bash
data/customer_support_tickets_200k.csv
```

---

## 4. Requisitos Previos

El proyecto fue construido y probado en entornos Windows utilizando Python 3.10 o superior.

Se recomienda ejecutar el sistema dentro de un entorno virtual aislado para evitar conflictos entre dependencias.

### Requisitos principales

* Python 3.10+
* pip
* Git
* Entorno virtual de Python
* Cuenta de Kaggle para descargar el dataset

---

## 5. Instalación del Proyecto

### 5.1. Clonar el repositorio

Abra una terminal en la ubicación deseada y ejecute el siguiente comando:

```bash
git clone https://github.com/SuUsuario/plataforma-itsm-ia.git
```

> Nota: Reemplace la URL anterior por la URL real de su repositorio.

### 5.2. Acceder al directorio del proyecto

```bash
cd plataforma-itsm-ia
```

### 5.3. Crear el entorno virtual

```bash
python -m venv venv
```

### 5.4. Activar el entorno virtual en Windows

```bash
.\venv\Scripts\activate
```

### 5.5. Instalar las dependencias

Con el entorno virtual activado, instale las librerías requeridas mediante el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 6. Entrenamiento de Modelos Analíticos

Antes de ejecutar la plataforma, es obligatorio generar los archivos serializados `.pkl`, los cuales contienen el vocabulario TF-IDF, los codificadores de etiquetas y los modelos predictivos entrenados.

Verifique que el dataset se encuentre en la siguiente ruta:

```bash
data/customer_support_tickets_200k.csv
```

Posteriormente, ejecute secuencialmente los scripts de entrenamiento.

### 6.1. Entrenar el modelo Baseline

El modelo Baseline utiliza Random Forest como algoritmo principal.

```bash
python src/models/train_baseline.py
```

### 6.2. Entrenar el modelo Retador

El modelo Retador utiliza XGBoost optimizado con SMOTE para mejorar el tratamiento del desbalance de clases.

```bash
python src/models/train_xgboost_pro.py
```

Al finalizar la ejecución, los artefactos analíticos serán generados dentro del directorio:

```bash
src/models/
```

---

## 7. Ejecución del Sistema

El sistema funciona bajo un esquema cliente-servidor. Por esta razón, es necesario ejecutar en paralelo el Backend y el Frontend en dos terminales independientes.

Asegúrese de tener activado el entorno virtual `venv` en ambas terminales.

---

### 7.1. Terminal 1: Inicialización de la API Backend

Esta terminal levanta el servidor asíncrono de FastAPI y carga los modelos predictivos en memoria.

```bash
uvicorn src.api.main:app --reload
```

La API estará disponible en la siguiente dirección:

```bash
http://127.0.0.1:8000
```

La documentación automática de la API, generada mediante Swagger UI, estará disponible en:

```bash
http://127.0.0.1:8000/docs
```

---

### 7.2. Terminal 2: Inicialización de la Interfaz Frontend

Esta terminal ejecuta la aplicación desarrollada en Streamlit, la cual permite interactuar con el sistema desde un panel visual.

```bash
streamlit run src/frontend/app.py
```

La aplicación abrirá automáticamente una pestaña en el navegador web. En caso de que no se abra, puede acceder manualmente a:

```bash
http://localhost:8501
```


---

## 8. Consideraciones Importantes

* El dataset no se incluye en el repositorio debido a su tamaño.
* El archivo CSV debe ubicarse obligatoriamente en la carpeta `data/`.
* Los modelos deben entrenarse antes de ejecutar el Backend y el Frontend.
* El Backend y el Frontend deben ejecutarse en terminales separadas.
* Se recomienda verificar que todos los archivos `.pkl` hayan sido generados correctamente antes de iniciar la plataforma.
* En caso de errores de carga de modelos, revise que los scripts de entrenamiento se hayan ejecutado sin fallos.

---

## 9. Tecnologías Utilizadas

El proyecto utiliza las siguientes tecnologías principales:

* Python
* FastAPI
* Streamlit
* Scikit-learn
* XGBoost
* Imbalanced-learn / SMOTE
* NLTK
* Pandas
* NumPy
* Joblib
* Uvicorn

---

## 10. Descripción Funcional del MVP

El MVP permite al usuario ingresar la descripción de un ticket de soporte técnico mediante una interfaz web. A partir de este texto, el sistema realiza un proceso de limpieza, transformación y vectorización del lenguaje natural, para posteriormente generar dos resultados predictivos:

1. La categoría técnica estimada del ticket.
2. El nivel de riesgo de incumplimiento del SLA.

Estos resultados permiten apoyar la toma de decisiones en procesos de mesa de ayuda, priorización de incidencias y asignación eficiente de recursos técnicos.

---

## 11. Autor

Trabajo de Fin de Máster desarrollado como parte de una propuesta de aplicación de Inteligencia Artificial en procesos de Gestión de Servicios de TI, ITSM.
