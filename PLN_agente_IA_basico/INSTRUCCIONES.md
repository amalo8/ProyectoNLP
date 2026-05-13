# Instrucciones de uso: agente con LangChain, Streamlit y OpenRouter

Esta practica muestra como crear un agente sencillo con LangChain que puede actualizar un fichero local usando criterios escritos en lenguaje natural.

La interfaz grafica esta hecha con Streamlit y la clave de API se obtiene desde OpenRouter. El modelo por defecto es `openrouter/free`, que intenta usar modelos gratuitos disponibles en OpenRouter.

## 1. Que hace esta aplicacion

La aplicacion permite:

- Ver el contenido de un fichero local llamado `data/notas.txt`.
- Escribir una instruccion en lenguaje natural.
- Pedir a un agente de LangChain que actualice ese fichero.
- Comprobar el resultado desde la propia interfaz.

Ejemplo de instruccion:

```text
Actualiza data/notas.txt anadiendo al final una nota personal: "Revisar los apuntes de matematicas antes del viernes".
```

## 2. Requisitos previos

Necesitas tener instalado:

- Conda, por ejemplo mediante Anaconda o Miniconda.
- Una clave de API de OpenRouter.
- Acceso a una terminal de Windows PowerShell.

Puedes crear una clave en:

```text
https://openrouter.ai/keys
```

## 3. Crear el entorno con Conda

Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
conda create -n agente-langchain python=3.11 -y
```

Activa el entorno:

```powershell
conda activate agente-langchain
```

## 4. Instalar dependencias

Con el entorno de Conda activado, instala las librerias:

```powershell
pip install -r requirements.txt
```

Las dependencias principales son:

- `streamlit`: crea la interfaz grafica.
- `langchain-classic`: proporciona el agente clasico.
- `langchain-core`: proporciona prompts y herramientas.
- `langchain-openai`: se usa como cliente compatible con el formato de chat de OpenRouter.
- `python-dotenv`: carga variables desde `.env`.

## 5. Configurar la clave de OpenRouter

Copia el fichero de ejemplo:

```powershell
Copy-Item .env.example .env
```

Abre el fichero `.env` y sustituye el texto por tu clave real:

```text
OPENROUTER_API_KEY=tu_clave_real_de_openrouter
```

No compartas este fichero ni lo subas a GitHub.

## 6. Ejecutar la aplicacion

Ejecuta:

```powershell
streamlit run app.py
```

Streamlit abrira una pagina web local. Normalmente sera:

```text
http://localhost:8501
```

## 7. Como usar la interfaz

La interfaz se ha reducido a una ventana de chatbot:

- Escribe una instruccion en la caja inferior del chat.
- El agente respondera en la conversacion.
- Si la instruccion requiere cambiar el fichero, el agente llamara a la herramienta `actualizar_fichero`.
- Para comprobar el resultado, abre manualmente `data/notas.txt` en Windows.

Ejemplo:

```text
Anade al final de data/notas.txt una nota para estudiar Python.
```

El modelo por defecto se configura en el codigo con `openrouter/free` y la temperatura con `0.0`, para mantener la interfaz lo mas simple posible.

## 8. Por que usamos `ChatOpenAI` con OpenRouter

OpenRouter permite usar distintos modelos desde una sola API. En este proyecto usamos `ChatOpenAI` como cliente compatible con el formato de chat, pero la clave y el endpoint son de OpenRouter.

En el codigo se configura:

```python
base_url="https://openrouter.ai/api/v1"
api_key=os.getenv("OPENROUTER_API_KEY")
```

El modelo por defecto es:

```text
openrouter/free
```

Ese valor pide a OpenRouter que seleccione automaticamente un modelo gratuito disponible. Tambien puedes elegir modelos concretos que tengan variante gratuita usando identificadores terminados en `:free`.

## 9. Como funciona el agente

El flujo general es:

```text
Usuario -> Streamlit -> Agente LangChain -> Herramienta Python -> Fichero local
```

Tambien tienes una imagen con el diagrama completo en:

```text
assets/flujo_trabajo_agente.svg
```

Y otro diagrama que relaciona cada parte conceptual del agente con las clases y funciones del codigo:

```text
assets/arquitectura_agente_ia.svg
```

El agente no modifica el fichero directamente. Para hacerlo debe usar una herramienta llamada:

```python
actualizar_fichero
```

Esta herramienta puede:

- Reemplazar todo el contenido.
- Anadir texto al final.
- Anadir texto al principio.

## 10. Seguridad del ejemplo

El agente solo puede modificar ficheros dentro de la carpeta:

```text
data/
```

Esto evita que pueda cambiar otros archivos del ordenador por error.

## 11. Problemas frecuentes

Si aparece un error relacionado con `OPENROUTER_API_KEY`, revisa que:

- Existe el fichero `.env`.
- La variable se llama exactamente `OPENROUTER_API_KEY`.
- La clave de OpenRouter es correcta.

Si Streamlit no arranca, revisa que:

- El entorno de Conda esta activado.
- Has instalado `requirements.txt`.
- Estas ejecutando el comando desde la carpeta del proyecto.
