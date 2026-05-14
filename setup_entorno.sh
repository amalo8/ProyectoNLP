#!/bin/bash
# ============================================================
#  setup_entorno.sh — Crea el entorno virtual y lo configura
#  Uso: bash setup_entorno.sh
# ============================================================

set -e   # Detener si cualquier comando falla

VENV_DIR=".venv"
PYTHON_MIN="3.10"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Configurador de Entorno — Dashboard Hotel  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ----- 1. Comprobar Python -----
if ! command -v python3 &>/dev/null; then
    echo "❌  Python3 no encontrado. Instálalo antes de continuar."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅  Python $PY_VERSION detectado."

# Versión mínima 3.10
MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo "❌  Se requiere Python >= 3.10. Tienes $PY_VERSION."
    exit 1
fi

# ----- 2. Crear entorno virtual -----
if [ -d "$VENV_DIR" ]; then
    echo "♻️   Entorno virtual ya existente en '$VENV_DIR'. Reutilizando..."
else
    echo "📦  Creando entorno virtual en '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# ----- 3. Activar entorno virtual -----
source "$VENV_DIR/bin/activate"
echo "✅  Entorno virtual activado."

# ----- 4. Actualizar pip -----
echo "⬆️   Actualizando pip..."
pip install --upgrade pip --quiet

# ----- 5. Instalar dependencias -----
echo "📥  Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt --quiet

echo ""
echo "✅  Instalación completada con éxito."

# ----- 6. Crear carpeta de datos si no existe -----
mkdir -p datos/procesados
echo "📁  Carpeta 'datos/procesados/' verificada."

# ----- 7. Crear .env si no existe -----
if [ ! -f ".env" ]; then
    echo "🔑  Creando archivo .env de ejemplo..."
    cat > .env << 'EOF'
# Pega aquí tu clave de OpenRouter
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
    echo "⚠️   Edita el archivo .env y añade tu OPENROUTER_API_KEY antes de arrancar."
else
    echo "🔑  Archivo .env ya existe."
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅  Entorno listo. Para arrancar la app:    ║"
echo "║                                              ║"
echo "║     source .venv/bin/activate                ║"
echo "║     streamlit run app.py                     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
