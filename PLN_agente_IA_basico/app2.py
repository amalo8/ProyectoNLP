"""
Aplicación educativa: Agente Versátil (Gestor de Tareas + Modificador de Textos)
con LangChain y Streamlit.

Objetivo:
- Mostrar cómo un LLM puede usar múltiples herramientas para modificar ficheros.
- Implementar una interfaz de dos columnas con tabla de datos (Pandas).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# ============================================================
# CONFIGURACIÓN DEL ENTORNO Y RUTAS
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DEFAULT_FILE = DATA_DIR / "notas.txt"
TAREAS_FILE = DATA_DIR / "tareas.txt"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELO_POR_DEFECTO = "openrouter/free"
TEMPERATURA_POR_DEFECTO = 0.0

# ============================================================
# 1. INTERFAZ GRÁFICA (UI)
# ============================================================

def main() -> None:
    """Interfaz de Streamlit para probar el agente."""

    st.set_page_config(
        page_title="Agente IA Multiherramienta",
        page_icon="🤖",
        layout="wide", # Usamos todo el ancho de la pantalla
    )

    # Estilos CSS para hacer la app más atractiva
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { border-radius: 8px; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

    preparar_ficheros()

    with st.sidebar:
        st.title("⚙️ Configuración")
        st.info("Este agente puede modificar ficheros de texto libre o gestionar tu lista estructurada de tareas.")
        if st.button("Restaurar ficheros de ejemplo"):
            restaurar_ficheros()
            st.success("Ficheros restaurados. Recarga la página.")
            st.rerun()

    st.title("🤖 Tu Agente Asistente")
    st.caption("Escribe una instrucción para modificar notas.txt o gestionar tus tareas.")

    # Dividimos la pantalla: Izquierda (Chatbot) | Derecha (Tabla de tareas)
    col_izq, col_der = st.columns([1, 1.2], gap="large")

    with col_izq:
        st.subheader("💬 Consola de Órdenes")
        instruccion = st.text_area(
            "Instrucción para el agente",
            placeholder="Ej: Añade la tarea de estudiar LangChain para el viernes",
            height=120
        )

        if st.button("Ejecutar agente", type="primary"):
            if not instruccion.strip():
                st.warning("Escribe primero una instrucción.")
            else:
                try:
                    with st.spinner("El agente está pensando y decidiendo qué herramienta usar..."):
                        agente = crear_agente(MODELO_POR_DEFECTO, TEMPERATURA_POR_DEFECTO)
                        respuesta = agente.invoke({"input": instruccion})
                        
                        st.success("¡Acción completada!")
                        st.markdown(f"**Respuesta del agente:**\n\n{respuesta['output']}")
                except Exception as exc:
                    st.error(f"No se pudo ejecutar el agente: {exc}")

    with col_der:
        st.subheader("📋 Tu Lista de Tareas")
        mostrar_tabla_tareas()

def mostrar_tabla_tareas():
    """Lee el fichero de tareas, lo muestra coloreado y lanza alertas de < 24h y < 6h."""
    if not TAREAS_FILE.exists() or TAREAS_FILE.stat().st_size == 0:
        st.info("No hay tareas registradas actualmente. ¡Pídele al agente que añada una!")
        return

    try:
        columnas = ["Estado", "Descripción", "Fecha Registro", "Fecha Límite"]
        datos = []
        alertas = [] # Aquí guardaremos las notificaciones
        ahora = datetime.now()

        lineas = TAREAS_FILE.read_text(encoding="utf-8").splitlines()
        
        for linea in lineas:
            partes = [p.strip() for p in linea.split("|")]
            if len(partes) == 4:
                estado_crudo = partes[0]
                descripcion = partes[1]
                fecha_limite_str = partes[3]
                
                estado_visual = "✅ Hecho" if estado_crudo == "[X]" else "⏳ Pendiente"
                datos.append([estado_visual, descripcion, partes[2], fecha_limite_str])

                # --- LÓGICA DE NOTIFICACIONES ---
                if estado_crudo == "[ ]": # Solo analizamos lo que NO está hecho
                    try:
                        limite_dt = datetime.strptime(fecha_limite_str, "%d/%m/%Y %H:%M")
                        diferencia = limite_dt - ahora
                        
                        if diferencia < timedelta(0):
                            alertas.append(f"❌ **CADUCADA:** La tarea '{descripcion}' tenía que estar lista el {fecha_limite_str}")
                        elif diferencia <= timedelta(hours=6):
                            # NUEVO: Si faltan 6 horas o menos, es muy urgente
                            alertas.append(f"🚨 **URGENTE:** La tarea '{descripcion}' caduca en menos de 6h ({fecha_limite_str})")
                        elif diferencia <= timedelta(days=1):
                            # Si faltan entre 6 y 24 horas, es un aviso normal
                            alertas.append(f"🔔 **AVISO:** La tarea '{descripcion}' caduca en menos de 24h ({fecha_limite_str})")
                    except ValueError:
                        pass # Ignoramos si la fecha no tiene el formato exacto

        if datos:
            df = pd.DataFrame(datos, columns=columnas)

            # --- LÓGICA DE COLORES DE LA TABLA ---
            def colorear_filas(row):
                # 1. Si está completada: VERDE
                if row['Estado'] == '✅ Hecho':
                    return ['background-color: #d4edda; color: #155724'] * len(row)
                
                # 2. Si está pendiente, calculamos la fecha para ver si es roja o amarilla
                try:
                    limite_dt = datetime.strptime(row['Fecha Límite'], "%d/%m/%Y %H:%M")
                    diferencia = limite_dt - datetime.now()
                    
                    if diferencia < timedelta(0):
                        # Caducada: Gris con texto tachado para diferenciarla
                        return ['background-color: #e2e3e5; color: #6c757d; text-decoration: line-through;'] * len(row)
                    elif diferencia <= timedelta(hours=6):
                        # NUEVO: Urgente (< 6h): ROJO
                        return ['background-color: #f8d7da; color: #721c24; font-weight: bold;'] * len(row)
                except ValueError:
                    pass
                
                # 3. Pendiente normal (> 6h o fecha irreconocible): AMARILLO
                return ['background-color: #fff3cd; color: #856404'] * len(row)

            # Aplicamos los estilos y mostramos la tabla
            df_estilizado = df.style.apply(colorear_filas, axis=1)
            st.dataframe(df_estilizado, use_container_width=True, hide_index=True)

            # --- MOSTRAR EL CENTRO DE NOTIFICACIONES ---
            if alertas:
                st.markdown("---")
                st.subheader("📬 Bandeja de Notificaciones")
                for alerta in alertas:
                    if "URGENTE" in alerta or "CADUCADA" in alerta:
                        st.error(alerta) # Sale en un recuadro rojo
                    else:
                        st.warning(alerta) # Sale en un recuadro amarillo
        else:
            st.warning("El archivo de tareas existe pero no tiene el formato correcto.")
    except Exception as e:
        st.error(f"Error al leer la tabla: {e}")

# ============================================================
# 2. CREACIÓN DEL AGENTE Y PROMPT
# ============================================================

def crear_agente(modelo: str, temperatura: float) -> AgentExecutor:
    llm = crear_llm(modelo, temperatura)
    prompt = crear_prompt()
    
    # LISTA DE TODAS LAS HERRAMIENTAS DISPONIBLES
    herramientas = [
        actualizar_fichero, 
        anadir_tarea, 
        listar_tareas_pendientes, 
        marcar_completada
    ]

    agente = create_tool_calling_agent(llm, herramientas, prompt)

    return AgentExecutor(
        agent=agente,
        tools=herramientas,
        verbose=True,
        handle_parsing_errors=True,
    )

def crear_llm(modelo: str, temperatura: float) -> ChatOpenAI:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("Falta OPENROUTER_API_KEY en el fichero .env.")

    return ChatOpenAI(
        model=modelo,
        temperature=temperatura,
        api_key=openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
    )

def crear_prompt():
    # Capturamos la fecha y hora exacta del momento en que se ejecuta
    fecha_actual = datetime.now().strftime("%A, %d/%m/%Y %H:%M")
    
    return ChatPromptTemplate.from_messages([
        (
            "system",
            f"Eres un agente versátil y ordenado. Tienes DOS capacidades principales:\n"
            "1. Modificar textos genéricos: Usa 'actualizar_fichero' para notas.txt u otros archivos libres.\n"
            "2. Gestor de Tareas: Usa 'anadir_tarea', 'listar_tareas_pendientes' y 'marcar_completada' para organizar tareas.\n\n"
            f"ATENCIÓN - RELOJ DEL SISTEMA: Hoy es {fecha_actual}.\n\n"
            "REGLAS CRÍTICAS:\n"
            "- Si el usuario quiere añadir una tarea, usa siempre 'anadir_tarea'.\n"
            "- Calcula la 'fecha_limite' usando el RELOJ DEL SISTEMA. Debes devolver la fecha límite SIEMPRE en formato exacto 'DD/MM/YYYY HH:MM' (ejemplo: 14/05/2026 17:00).\n"
            "- Si te piden completar/marcar una tarea, DEBES ejecutar 'listar_tareas_pendientes' primero para encontrar el ID correcto."
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

# ============================================================
# 3. ESQUEMAS DE LAS HERRAMIENTAS (PYDANTIC)
# ============================================================

class ActualizarFicheroArgs(BaseModel):
    ruta_relativa: str = Field(description="Ruta del fichero dentro de la carpeta data.")
    accion: Literal["reemplazar", "anadir_al_final", "anadir_al_principio"] = Field(description="Accion a aplicar.")
    contenido: str = Field(description="Contenido a escribir.")

class AnadirTareaArgs(BaseModel):
    descripcion: str = Field(description="Descripción de la tarea a realizar.")
    fecha_limite: str = Field(description="Fecha y hora límite calculada en formato estricto 'DD/MM/YYYY HH:MM'.")
class MarcarCompletadaArgs(BaseModel):
    id_tarea: int = Field(description="El número de ID de la tarea a completar.")

# ============================================================
# 4. HERRAMIENTAS (LO QUE EL AGENTE PUEDE HACER)
# ============================================================

@tool("actualizar_fichero", args_schema=ActualizarFicheroArgs)
def actualizar_fichero(ruta_relativa, accion, contenido):
    """Herramienta para textos libres. No la uses para tareas estructuradas."""
    ruta = resolver_ruta_segura(ruta_relativa)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    texto_actual = ruta.read_text(encoding="utf-8") if ruta.exists() else ""

    if accion == "reemplazar": nuevo_texto = contenido
    elif accion == "anadir_al_final": nuevo_texto = texto_actual.rstrip() + "\n" + contenido
    elif accion == "anadir_al_principio": nuevo_texto = contenido.rstrip() + "\n" + texto_actual
    else: raise ValueError(f"Accion no soportada: {accion}")

    ruta.write_text(nuevo_texto, encoding="utf-8")
    return f"Fichero {ruta.name} actualizado. Acción: {accion}"

@tool("anadir_tarea", args_schema=AnadirTareaArgs)
def anadir_tarea(descripcion: str, fecha_limite: str):
    """Añade una nueva tarea estructurada a tareas.txt."""
    # Ahora guardamos la fecha de registro CON hora (HH:MM)
    fecha_registro = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    nueva_linea = f"[ ] | {descripcion} | {fecha_registro} | {fecha_limite}\n"
    
    with open(TAREAS_FILE, "a", encoding="utf-8") as f:
        f.write(nueva_linea)
    return f"Tarea '{descripcion}' guardada exitosamente."

@tool("listar_tareas_pendientes")
def listar_tareas_pendientes():
    """Lee tareas.txt y devuelve el ID (número de línea) y el contenido al agente."""
    if not TAREAS_FILE.exists(): return "No hay tareas registradas."
    
    lineas = TAREAS_FILE.read_text(encoding="utf-8").splitlines()
    resultado = "IDs y Tareas:\n"
    for i, linea in enumerate(lineas, 1):
        resultado += f"ID {i}: {linea}\n"
    return resultado

@tool("marcar_completada", args_schema=MarcarCompletadaArgs)
def marcar_completada(id_tarea: int):
    """Marca con [X] la tarea indicada por su número de ID."""
    if not TAREAS_FILE.exists(): return "El archivo de tareas no existe."
    
    lineas = TAREAS_FILE.read_text(encoding="utf-8").splitlines()
    if 1 <= id_tarea <= len(lineas):
        lineas[id_tarea-1] = lineas[id_tarea-1].replace("[ ]", "[X]")
        TAREAS_FILE.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        return f"Éxito: Tarea ID {id_tarea} marcada como completada."
    return f"Error: No existe el ID {id_tarea}."

# ============================================================
# 5. FUNCIONES AUXILIARES DE SEGURIDAD Y CONFIGURACIÓN
# ============================================================

def preparar_ficheros() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not DEFAULT_FILE.exists(): restaurar_ficheros()

def restaurar_ficheros() -> None:
    # Restauramos notas.txt
    DEFAULT_FILE.write_text(
        "Lista de notas libres:\n- Estas notas no tienen estructura.\n",
        encoding="utf-8",
    )
    # Vaciamos/creamos tareas.txt limpio
    TAREAS_FILE.write_text(
        "[ ] | Exponer el agente en clase | 13/05/2026 11:05 | 13/05/2026 11:30\n", 
        encoding="utf-8"
    )

def resolver_ruta_segura(ruta_relativa: str) -> Path:
    ruta_limpia = ruta_relativa.strip().replace("\\", "/")
    if ruta_limpia.startswith("data/"): ruta_limpia = ruta_limpia.removeprefix("data/")
    ruta = (DATA_DIR / ruta_limpia).resolve()
    if DATA_DIR not in ruta.parents and ruta != DATA_DIR:
        raise ValueError("La ruta indicada esta fuera de la carpeta data.")
    return ruta

# ============================================================
# ARRANQUE DE LA APLICACIÓN
# ============================================================

if __name__ == "__main__":
    main()