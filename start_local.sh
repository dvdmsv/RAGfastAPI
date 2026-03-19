#!/bin/bash
echo "Parando contenedores de Docker si están activos..."
docker compose down || true

echo "Activando entorno virtual..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

echo "Instalando dependencias (puede tardar un poco)..."
pip install -r requirements.txt

echo "Configurando variables de entorno para LM Studio..."
export OPENAI_API_BASE=http://localhost:1234/v1
export OPENAI_API_KEY=lm-studio
export LLM_MODEL=gpt-3.5-turbo

echo "Iniciando FastAPI (Backend)..."
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend_local.log 2>&1 &

echo "Iniciando Streamlit (Frontend)..."
nohup streamlit run frontend.py --server.port 8501 > frontend_local.log 2>&1 &

echo "================================================="
echo "✅ Backend ejecutándose en: http://localhost:8000"
echo "✅ Frontend ejecutándose en: http://localhost:8501"
echo "================================================="
echo "Nota: Si los puertos ya están en uso, puede fallar."
echo "Para matar los procesos locales más tarde:"
echo "pkill -f uvicorn && pkill -f streamlit"
