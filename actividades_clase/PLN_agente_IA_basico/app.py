import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

# --- IMPORTS DE LANGCHAIN ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Cargar las variables ocultas del archivo .env
load_dotenv()

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard Hotelero NLP", page_icon="🏨", layout="wide")
st.title("🏨 Cuadro de Mandos y Agente IA de Calidad")
st.markdown("Analizador inteligente de sentimiento y tópicos con arquitectura LangChain.")

# ==========================================
# 2. CARGA DE DATOS (Con caché)
# ==========================================
@st.cache_data
def cargar_datos():
    df_info = pd.read_csv('datos/procesados/df_hoteles_vlc_info.csv')
    df_comentarios = pd.read_csv('datos/procesados/df_comentarios_final_topics.csv')
    return df_info, df_comentarios

df_info, df_comentarios = cargar_datos()

# ==========================================
# 3. SELECTOR Y FILTRADO (ROBUSTO)
# ==========================================
lista_hoteles = df_info['nombre_del_hotel'].unique()
hotel_seleccionado = st.selectbox("📌 Selecciona un Hotel para analizar:", lista_hoteles)

# Filtrado ignorando mayúsculas/minúsculas
info_hotel = df_info[df_info['nombre_del_hotel'].str.lower() == hotel_seleccionado.lower()].iloc[0]
comentarios_hotel = df_comentarios[df_comentarios['nombre_del_hotel'].str.lower() == hotel_seleccionado.lower()]


# ==========================================
# 4. FUNCIONES BASE (lógica pura)
# ==========================================
def _consultar_metricas():
    return info_hotel.to_dict()

def _analizar_comentarios():
    def extraer(col):
        todos = []
        for c in comentarios_hotel[col].dropna():
            temas = str(c).split(" - ")
            todos.extend([t for t in temas if t != "Otro Tema / Mixto / Vacío"])
        return pd.Series(todos).value_counts().head(3).index.tolist()
    return {
        "fortalezas": extraer('tema_positivo'),
        "debilidades": extraer('tema_negativo')
    }

def _obtener_evidencia_real(categoria, tipo):
    """Filtra y devuelve ejemplos reales de texto para una categoría específica."""
    col_tema = 'tema_positivo' if tipo == 'positivo' else 'tema_negativo'
    col_texto = 'positivo' if tipo == 'positivo' else 'negativo'
   
    # Buscamos filas donde el tema contenga la categoría (ej: 'Confort')
    muestras = comentarios_hotel[comentarios_hotel[col_tema].str.contains(categoria, na=False, case=False)]
   
    if muestras.empty:
        return f"No se han encontrado quejas o alabanzas específicas sobre {categoria}."
   
    # Devolvemos los 6 ejemplos más representativos
    ejemplos = muestras[col_texto].head(6).tolist()
    return "\n".join([f"- {txt}" for txt in ejemplos])

# ==========================================
# 5. SISTEMA DE PESTAÑAS
# ==========================================
tab1, tab2 = st.tabs(["📊 Análisis de Puntuaciones", "🤖 Consultor IA LangChain"])

# ------------------------------------------
# PESTAÑA 1: DASHBOARD VISUAL
# ------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    col1.metric(label="⭐ Nota Media Booking", value=f"{info_hotel['nota_media_resenas']} / 10")
    col2.metric(label="📝 Volumen de Reseñas Analizadas", value=len(comentarios_hotel))

    st.markdown("<br>", unsafe_allow_html=True)

    categorias_base = ['Personal', 'Confort', 'WiFi', 'Instalaciones', 'Calidad/Precio', 'Limpieza', 'Ubicación']
    columnas_notas = ['nota_personal', 'nota_confort', 'nota_wifi', 'nota_instalaciones_servicios', 'nota_calidad_precio', 'nota_limpieza', 'nota_ubicacion']
    cat_validas, punt_hotel, med_globales, cat_faltantes = [], [], [], []

    for cat, col in zip(categorias_base, columnas_notas):
        nota = info_hotel[col]
        if pd.isna(nota):
            cat_faltantes.append(cat)
        else:
            cat_validas.append(cat)
            punt_hotel.append(nota)
            med_globales.append(round(df_info[col].mean(), 1))

    if cat_faltantes:
        st.warning(f"⚠️ Categorías sin datos: **{', '.join(cat_faltantes)}**")

    if len(cat_validas) >= 3:
        min_abs = min(min(punt_hotel), min(med_globales))
        inicio_rango = int(min_abs) if min_abs >= 5 else 0
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=med_globales + [med_globales[0]], theta=cat_validas + [cat_validas[0]],
            fill='toself', fillcolor='rgba(255, 65, 54, 0.15)',
            line=dict(color='rgba(255, 65, 54, 0.5)', width=1.5), name='Media del Sector'
        ))
        fig.add_trace(go.Scatterpolar(
            r=punt_hotel + [punt_hotel[0]], theta=cat_validas + [cat_validas[0]],
            fill='toself', fillcolor='rgba(31, 119, 180, 0.5)',
            line=dict(color='#1f77b4', width=3), marker=dict(size=8), name=hotel_seleccionado
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[inicio_rango, 10], dtick=0.5)),
                          showlegend=True, margin=dict(l=80, r=80, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# PESTAÑA 2: EL AGENTE LANGCHAIN
# ------------------------------------------
with tab2:
    st.subheader(f"🤖 Consultor Estratégico IA: {hotel_seleccionado}")

    # --- HERRAMIENTAS (TOOLS) ---
    @tool
    def consultar_metricas_seleccionado():
        """Consulta las notas numéricas actuales del hotel seleccionado."""
        return _consultar_metricas()

    @tool
    def analizar_comentarios_nlp():
        """Analiza los tópicos positivos y negativos más frecuentes del hotel."""
        return _analizar_comentarios()

    @tool
    def profundizar_en_comentarios_reales(categoria: str, tipo: str):
        """
        Lee ejemplos reales de reseñas para entender el porqué de una nota.
        'categoria' debe ser el tema (ej: 'Confort', 'Limpieza').
        'tipo' debe ser 'positivo' o 'negativo'.
        """
        return _obtener_evidencia_real(categoria, tipo)

    # Pre-visualización
    top_data = _analizar_comentarios()
    col_a, col_b = st.columns(2)
    with col_a:
        st.success(f"✅ **Fortalezas:** {', '.join(top_data['fortalezas'])}")
    with col_b:
        st.error(f"❌ **Debilidades:** {', '.join(top_data['debilidades'])}")

    if st.button("🚀 Ejecutar Consultoría"):
        with st.spinner('🕵️ Investigando a fondo las reseñas...'):
            clave = os.getenv("OPENROUTER_API_KEY")
           
            # Priorizamos modelos potentes para que sigan mejor las instrucciones complejas
            modelos_prueba = [
                "nvidia/nemotron-3-super-120b-a12b:free",
                "meta-llama/llama-3-8b-instruct:free",
                "google/gemma-2-9b-it:free"
            ]
           
            exito = False
            for m in modelos_prueba:
                try:
                    llm = ChatOpenAI(model=m, api_key=clave, base_url="https://openrouter.ai/api/v1", temperature=0.1)
                    tools = [consultar_metricas_seleccionado, analizar_comentarios_nlp, profundizar_en_comentarios_reales]
                   
                    # --- EL PROMPT DE HIERRO ---
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", f"""Eres un Auditor de Calidad de Hoteles. Tu informe sobre el {hotel_seleccionado} será leído por la directiva y DEBE ser técnico y basado en pruebas.
                       
                        PROCESO OBLIGATORIO (Paso a paso):
                        1. Llama a 'consultar_metricas_seleccionado' para ver las notas.
                        2. Llama a 'analizar_comentarios_nlp' para identificar los 3 problemas principales.
                        3. Para CADA uno de esos 3 problemas, DEBES LLAMAR OBLIGATORIAMENTE a 'profundizar_en_comentarios_reales' (tipo='negativo').
                        4. Si no usas la herramienta de profundizar, tu informe será rechazado.
                       
                        ESTRUCTURA DEL INFORME FINAL:
                        - Diagnóstico Numérico: Breve resumen de las notas.
                         IMPORTANTÍSIMO, OBLIGATORIO:
                        - Análisis de Evidencias: Por cada problema detectado, cita TEXTUALMENTE entre comillas al menos una frase real de los clientes que has leído.
                        - Plan de Acción: Soluciones concretas a esos testimonios.
                       
                        PROHIBIDO: Usar frases genéricas como 'mejorar el servicio' o 'el personal es amable'. Di qué pasa exactamente."""),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                   
                    agent = create_tool_calling_agent(llm, tools, prompt)
                    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                    # --- CAMBIO EN LA CONSULTA: Pedimos evidencia explícitamente ---
                    consulta = (
                        f"Realiza una auditoría completa del {hotel_seleccionado}. "
                        "Identifica los fallos críticos y utiliza la herramienta de profundizar para "
                        "extraer testimonios reales. Tu respuesta DEBE incluir citas directas de clientes."
                    )

                    respuesta = executor.invoke({"input": consulta})

                    st.markdown("---")
                    st.markdown(respuesta["output"])
                    st.caption(f"🧠 Cerebro utilizado: {m}")
                    exito = True
                    break
                except Exception as e:
                    st.warning(f"Fallo con {m}. Reintentando...")
                    continue

            if not exito:
                st.error("Servidores saturados. Reintenta en unos segundos.")