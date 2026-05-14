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
    # Asegúrate de que las rutas sean las correctas
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

st.divider()

# ==========================================
# 4. SISTEMA DE PESTAÑAS
# ==========================================
tab1, tab2 = st.tabs(["📊 Análisis de Puntuaciones", "🤖 Consultor IA LangChain"])

# ------------------------------------------
# PESTAÑA 1: DASHBOARD VISUAL (GRÁFICO DE ARAÑA)
# ------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    col1.metric(label="⭐ Nota Media Booking", value=f"{info_hotel['nota_media_resenas']} / 10")
    col2.metric(label="📝 Volumen de Reseñas Analizadas", value=len(comentarios_hotel))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Lógica del Gráfico de Araña con Benchmarking ---
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
        # Capa Fondo: Media Sector
        fig.add_trace(go.Scatterpolar(
            r=med_globales + [med_globales[0]], theta=cat_validas + [cat_validas[0]],
            fill='toself', fillcolor='rgba(255, 65, 54, 0.15)', 
            line=dict(color='rgba(255, 65, 54, 0.5)', width=1.5), name='Media del Sector'
        ))
        # Capa Frente: Hotel
        fig.add_trace(go.Scatterpolar(
            r=punt_hotel + [punt_hotel[0]], theta=cat_validas + [cat_validas[0]],
            fill='toself', fillcolor='rgba(31, 119, 180, 0.5)', 
            line=dict(color='#1f77b4', width=3), marker=dict(size=8), name=hotel_seleccionado
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[inicio_rango, 10], dtick=0.5)),
            showlegend=True, margin=dict(l=80, r=80, t=20, b=20)
        )
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
        return info_hotel.to_dict()

    @tool
    def analizar_comentarios_nlp():
        """Analiza los tópicos positivos y negativos más frecuentes del hotel."""
        def extraer(col):
            todos = []
            for c in comentarios_hotel[col].dropna():
                temas = str(c).split(" - ")
                todos.extend([t for t in temas if t != "Otro Tema / Mixto / Vacío"])
            return pd.Series(todos).value_counts().head(3).index.tolist()
        return {"fortalezas": extraer('tema_positivo'), "debilidades": extraer('tema_negativo')}

    # Pre-visualización de tópicos (para dar contexto al usuario)
    top_data = analizar_comentarios_nlp()
    col_a, col_b = st.columns(2)
    with col_a:
        st.success(f"✅ **Fortalezas:** {', '.join(top_data['fortalezas'])}")
    with col_b:
        st.error(f"❌ **Debilidades:** {', '.join(top_data['debilidades'])}")

    if st.button("🚀 Ejecutar Consultoría"):
        with st.spinner('El Agente LangChain está razonando...'):
            clave = os.getenv("OPENROUTER_API_KEY")
            
            # LISTA DE MODELOS DE RESPALDO (Evita 404 y 429)
            modelos_prueba = [
                "nvidia/nemotron-3-super-120b-a12b:free",
                "meta-llama/llama-3.2-3b-instruct:free",
                "meta-llama/llama-3-8b-instruct:free",
                "mistralai/mistral-7b-instruct:free"
            ]

            exito = False
            for m in modelos_prueba:
                try:
                    llm = ChatOpenAI(
                        model=m, api_key=clave,
                        base_url="https://openrouter.ai/api/v1", temperature=0.3
                    )
                    tools = [consultar_metricas_seleccionado, analizar_comentarios_nlp]
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", f"Eres un Consultor Senior. Analiza el hotel {hotel_seleccionado}."),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    agent = create_tool_calling_agent(llm, tools, prompt)
                    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                    
                    respuesta = executor.invoke({"input": f"Dime los 2 puntos de mejora más urgentes para {hotel_seleccionado}."})
                    
                    st.markdown("---")
                    st.markdown(respuesta["output"])
                    st.caption(f"🧠 Cerebro utilizado: {m}")
                    exito = True
                    break
                except Exception as e:
                    st.warning(f"Fallo con {m}. Reintentando con el siguiente...")
                    continue
            
            if not exito:
                st.error("Servidores de OpenRouter saturados. Reintenta en 30 segundos.")