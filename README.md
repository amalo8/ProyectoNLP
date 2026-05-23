# 🏨 Análisis de Reseñas de Hoteles mediante PLN

Trabajo final de la asignatura **Procesado del Lenguaje Natural** del Máster en Ciencia de Datos de la Universitat de València (Curso 2025/2026).

Este proyecto consiste en el desarrollo de un sistema avanzado de análisis de opiniones para 12 hoteles de la ciudad de Valencia, aplicando técnicas de Procesado del Lenguaje Natural (PLN). A partir de un corpus masivo de **58.720 reseñas** recopiladas mediante *web scraping*, el estudio combina el análisis textual y estadístico para extraer inteligencia de negocio de alto valor estratégico para el sector hotelero.

La idea central del trabajo es superar las limitaciones de las valoraciones numéricas tradicionales, profundizando en el componente emocional del texto libre redactado por los huéspedes. El análisis conjunto de los comentarios y las puntuaciones permite identificar con precisión qué servicios específicos impactan más en la percepción global del cliente, así como diagnosticar problemas operativos concretos descritos en las reseñas para proponer mejoras accionables.

## ✨ Objetivos del Proyecto

* **Predicción de la satisfacción:** Clasificación del sentimiento del cliente y puntuación del hotel.
* **Análisis por nacionalidad:** Identificación de la exigencia y de los aspectos más relevantes según el país de origen.
* **Extracción de tópicos (ABSA):** Análisis de sentimiento basado en aspectos y validación estadística.
* **Agente Inteligente:** Implementación de un cuadro de mandos interactivo mediante LangChain.

## 📁 Estructura del Repositorio

A continuación se detalla el contenido de este repositorio:

```text
├── datos/                              # Datos crudos extraídos y conjuntos procesados
├── figuras/                            # Gráficos y visualizaciones generadas en el EDA
├── agente/                             # Código del agente inteligente (LangChain)
├── 01_preprocesado.ipynb               # Limpieza, normalización y traducción de reseñas
├── 02_union_datos.ipynb                # Unión de CSVs de los distintos hoteles
├── 03_analisis_exploratorio.ipynb      # EDA: distribución de notas y análisis de sentimiento
├── 04_prediccion_satisfaccion.ipynb    # Fine-tuning de BETO para clasificación de reseñas
├── 05_analisis_nacionalidad.ipynb      # TF-IDF, Log Odds y análisis por nacionalidad
├── 06_absa_topic_modeling.ipynb        # ABSA Zero-Shot con embeddings multilingües
├── 07_indices_correlaciones.ipynb      # Índices PLN, Time-Decay y correlaciones de Pearson
├── 08_agente_inteligente.ipynb         # Agente LangChain: auditoría y monitorización
├── app.py                              # Aplicación Streamlit del dashboard interactivo
├── requirements.txt                    # Dependencias necesarias para ejecutar el proyecto
└── README.md                           # Este archivo
```

## 🚀 Ejecución del agente
El agente se encuentra disponible en tiempo real a través de el siguiente enlace en *Streamlit Community Cloud*: https://dashboard-hotelero-agente-e9harbktzrwg46rbjpxxsr.streamlit.app/

## 👩‍💻 Autoras

Adriana Marí López, María de los Ángeles Díaz Castro, Florencia Pellegrini, Irene Barba La Orden, Ana Blasco Vega, Lucía Benages Guijarro.
