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
├── actividades_clase/          # Pruebas y ejercicios previos (Agentes, Topic Modeling)
├── agente/                     # Código de despliegue del agente inteligente (LangChain)
│   ├── app.py                  # Aplicación principal del cuadro de mandos (Streamlit)
│   └── requirements.txt        # Dependencias específicas del agente
├── datos/                      # Almacenamiento de datos del proyecto
│   ├── procesados/             # Datasets finales tras limpieza, traducción y modelado
│   └── scraping_hoteles/       # Datos crudos extraídos de cada hotel original
├── figuras/                    # Gráficos generados en los análisis 
├── objetivo1/                  # Scripts del 1º Objetivo
│   ├── Analisis_exploratorio.ipynb # EDA (Distribución de notas y sentimiento)
│   └── predictor_nota.ipynb    # Fine-tuning de BETO para predicción de satisfacción
├── objetivo2/                  # Scripts del 2º Objetivo
│   └── nacionalidades.ipynb    # Análisis por nacionalidad (TF-IDF, Log Odds)
├── objetivo3/                  # Scripts del 3º Objetivo
│   ├── ABSA.ipynb              # Análisis de Sentimiento Basado en Aspectos (Zero-Shot)
│   └── prueba_beto.ipynb       # Pruebas de validación
├── preprocesado/               # Scripts de limpieza inicial y consolidación
│   ├── preprocesado.ipynb      # Preprocesado dataset comentarios
│   ├── tokenizacion.ipynb      # Tokenización y lematización comentarios
│   ├── traduccion.ipynb        # Traducción de comentarios
|   ├── topic_modeling.ipynb    # Extracción de tópicos 
│   └── union_csv.ipynb         # Unión de los datos scrapeados
├── objetivos.txt               # Archivo de brainstorming y objetivos del proyecto
└── README.md                   # Este archivo
```

## 🚀 Ejecución del agente
El agente se encuentra disponible en tiempo real a través de el siguiente enlace en *Streamlit Community Cloud*: https://dashboard-hotelero-agente-e9harbktzrwg46rbjpxxsr.streamlit.app/

## 👩‍💻 Autoras

Adriana Marí López, María de los Ángeles Díaz Castro, Florencia Pellegrini, Irene Barba La Orden, Ana Blasco Vega, Lucía Benages Guijarro.
