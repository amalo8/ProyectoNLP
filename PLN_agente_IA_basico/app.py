"""
Aplicacion educativa: agente sencillo con LangChain + Streamlit.

Objetivo:
- Mostrar como un LLM puede usar una "herramienta" para modificar un fichero local.
- Mantener el codigo claro para alumnos que empiezan desde cero.

Para ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal


# ============================================================
# 1. INTERFAZ GRÁFICA
# ============================================================

# Streamlit permite crear la interfaz web de la aplicación.
import streamlit as st


# ============================================================
# 2. CONFIGURACIÓN DEL ENTORNO
# ============================================================

# load_dotenv carga variables de entorno desde un fichero .env.
# Por ejemplo: OPENROUTER_API_KEY=...
from dotenv import load_dotenv


# ============================================================
# 3. MODELO LLM
# ============================================================

# ChatOpenAI es el cliente de chat compatible con APIs tipo OpenAI.
# Aquí se usará con OpenRouter configurando api_key y base_url.
from langchain_openai import ChatOpenAI


# ============================================================
# 4. PROMPT DEL AGENTE
# ============================================================

# ChatPromptTemplate construye el prompt con mensajes separados.
# MessagesPlaceholder reserva espacio para pasos intermedios del agente.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # libreria para confeccionar instrucciones internas del agente


# ============================================================
# 5. HERRAMIENTAS DEL AGENTE
# ============================================================

# tool convierte una función Python normal en una herramienta para el agente.
from langchain_core.tools import tool

# BaseModel y Field definen y documentan los argumentos de la herramienta.
from pydantic import BaseModel, Field


# ============================================================
# 6. EJECUCIÓN DEL AGENTE
# ============================================================

# create_tool_calling_agent crea un agente capaz de llamar herramientas.
# AgentExecutor ejecuta el ciclo: entrada -> decisión -> tool -> respuesta.
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent




# Cargamos variables de entorno desde un fichero .env si existe.
# Por ejemplo: OPENROUTER_API_KEY=sk-or-...
load_dotenv()


# Solo permitimos tocar ficheros dentro de esta carpeta.
# Esto evita que el agente pueda modificar cualquier archivo del ordenador.
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DEFAULT_FILE = DATA_DIR / "notas.txt"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELO_POR_DEFECTO = "openrouter/free"
TEMPERATURA_POR_DEFECTO = 0.0 # cuanto + alto + probable es que se invente cualquier cosa


# ============================================================
# 1. INTERFAZ: el usuario escribe una instrucción
# ============================================================

def main() -> None:
    """Interfaz de Streamlit para probar el agente."""

    st.set_page_config(
        page_title="Agente con LangChain",
        layout="centered",
    )

    preparar_fichero_demo() # funcion para crear el data/notas.txt con la info predefinida

    st.title("Agente sencillo con LangChain")
    st.caption("Escribe una instruccion y el agente actualizara data/notas.txt si procede.")

    instruccion = st.text_area(
        "Instrucción para el agente",
        placeholder="Ejemplo: Añade al final de data/notas.txt una nota para estudiar Python.",   # En instruccion es donde el usuario escribe lo que quiere que el agente haga. El placeholder es un ejemplo que se muestra cuando el campo está vacío para guiar al usuario.
    )

    if st.button("Ejecutar agente"):
        if not instruccion.strip():
            st.warning("Escribe primero una instrucción.")
            return

        try:
            agente = crear_agente(
                modelo=MODELO_POR_DEFECTO,
                temperatura=TEMPERATURA_POR_DEFECTO,
            )

            respuesta = agente.invoke({"input": instruccion}) 
            # invoke es el método que ejecuta el agente. 
            # Le pasamos un diccionario con la clave "input" porque en el prompt definimos un mensaje 
            # human con "{input}". El agente procesa la instrucción, decide si usar la herramienta actualizar_fichero, 
            # y devuelve una respuesta final que se muestra al usuario.

            st.subheader("Respuesta del agente")
            st.write(respuesta["output"])

        except Exception as exc:
            st.error(f"No se pudo ejecutar el agente: {exc}")

    if st.button("Restaurar fichero de ejemplo"):
        restaurar_fichero_demo()
        st.success("Fichero restaurado correctamente.")


# ============================================================
# 2. CREACIÓN DEL AGENTE
# ============================================================

def crear_agente(modelo: str, temperatura: float) -> AgentExecutor:
    """
    Construye el agente.

    El agente necesita:
    1. Un modelo LLM.
    2. Un prompt con instrucciones.
    3. Una lista de herramientas.
    """

    llm = crear_llm(modelo, temperatura)
    prompt = crear_prompt()
    herramientas = [actualizar_fichero]

    agente = create_tool_calling_agent(llm, herramientas, prompt)

    return AgentExecutor(  # retorna un objeto de la clase AgenteExecutor que se encarga de ejecutar el ciclo del agente.
        agent=agente,
        tools=herramientas,
        verbose=True,
        handle_parsing_errors=True,
    )


# ============================================================
# 3. MODELO LLM
# ============================================================

def crear_llm(modelo: str, temperatura: float) -> ChatOpenAI:
    """Configura el modelo de lenguaje que tomará las decisiones."""

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("Falta OPENROUTER_API_KEY en el fichero .env.")

    return ChatOpenAI(
        model=modelo,
        temperature=temperatura,
        api_key=openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Agente educativo LangChain Streamlit",
        },
    )


# ============================================================
# 4. PROMPT: instrucciones que recibe el agente
# ============================================================

# Define las instrucciones generales del agente

def crear_prompt():

    return ChatPromptTemplate.from_messages([

        # Comportamiento general del agente
        (
            "system",
            "Eres un agente que modifica ficheros dentro de la carpeta data. No permitas crear subcarpetas. "
            "Usa la herramienta actualizar_fichero para hacer cambios reales."
        ),

        # Mensaje del usuario
        ("human", "{input}"),   # aquí el agente recibirá la instrucción del usuario. El "{input}" se reemplazará por el texto que el usuario escriba en la interfaz.

        # Memoria interna del agente
        MessagesPlaceholder(variable_name="agent_scratchpad"), # este espacio se reserva para que el agente pueda escribir sus pensamientos intermedios, como "Voy a usar la herramienta actualizar_fichero con estos argumentos...". Esto es útil para entender qué decisiones toma el agente durante su proceso de razonamiento.

    ])


# ============================================================
# 5. ESQUEMA DE ENTRADA DE LA HERRAMIENTA
# ============================================================

class ActualizarFicheroArgs(BaseModel):
    """
    Argumentos que el LLM debe generar para usar la herramienta.

    Esto no modifica nada: solo define qué datos son válidos.
    """

    ruta_relativa: str = Field(
        description="Ruta del fichero dentro de la carpeta data." # El texto de la descripción es lo que el LLM ve para entender qué debe escribir aquí.
    )

    accion: Literal["reemplazar", "anadir_al_final", "anadir_al_principio"] = Field(
        description="Accion que se aplicara al fichero."
    )

    contenido: str = Field(
        description="Contenido que se escribira en el fichero."
    )


# ============================================================
# 6. HERRAMIENTA: acción real que puede ejecutar el agente
# ============================================================

@tool("actualizar_fichero", args_schema=ActualizarFicheroArgs)
def actualizar_fichero(ruta_relativa, accion, contenido):
    """
    Tool de LangChain.

    El LLM decide usarla, pero Python es quien realmente modifica el fichero.
    """

    ruta = resolver_ruta_segura(ruta_relativa)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    texto_actual = ruta.read_text(encoding="utf-8") if ruta.exists() else ""

    if accion == "reemplazar":
        nuevo_texto = contenido
    elif accion == "anadir_al_final":
        nuevo_texto = texto_actual.rstrip() + "\n" + contenido
    elif accion == "anadir_al_principio":
        nuevo_texto = contenido.rstrip() + "\n" + texto_actual
    else:
        raise ValueError(f"Accion no soportada: {accion}")

    ruta.write_text(nuevo_texto, encoding="utf-8")

    return (
        f"Fichero actualizado correctamente.\n"
        f"Ruta: data/{ruta.name}\n"
        f"Accion aplicada: {accion}\n"
        f"Caracteres finales: {len(nuevo_texto)}"
    )


# ============================================================
# 7. FUNCIONES AUXILIARES DEL FICHERO
# ============================================================

def preparar_fichero_demo() -> None:
    """Crea la carpeta y el fichero de ejemplo si no existen."""

    DATA_DIR.mkdir(exist_ok=True)
    if not DEFAULT_FILE.exists():
        restaurar_fichero_demo()


def restaurar_fichero_demo() -> None:
    """Restaura el fichero de ejemplo a su contenido inicial."""

    DEFAULT_FILE.write_text(
        "Lista inicial de notas personales:\n"
        "- Organizar mis apuntes de inteligencia artificial\n"
        "- Revisar dudas antes de la proxima clase\n",
        encoding="utf-8",
    )


def resolver_ruta_segura(ruta_relativa: str) -> Path:
    """Evita que el agente escriba fuera de la carpeta data."""

    ruta_limpia = ruta_relativa.strip().replace("\\", "/")

    if ruta_limpia.startswith("data/"):
        ruta_limpia = ruta_limpia.removeprefix("data/")

    ruta = (DATA_DIR / ruta_limpia).resolve()

    if DATA_DIR not in ruta.parents and ruta != DATA_DIR:
        raise ValueError("La ruta indicada esta fuera de la carpeta data.")

    return ruta


# ============================================================
# 8. ARRANQUE DE LA APLICACIÓN
# ============================================================

if __name__ == "__main__":
    main()