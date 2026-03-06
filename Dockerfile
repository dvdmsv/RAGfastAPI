# Usamos la versión oficial de Python 3.12 ligera
FROM python:3.12-slim

# Evitamos que Python escriba archivos temporales (.pyc) y forzamos el log en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero el requirements.txt (esto optimiza el caché de Docker)
COPY requirements.txt .

# Instalamos todas tus dependencias congeladas
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de tu código al contenedor
COPY . .

# Exponemos los puertos que usaremos (8000 para FastAPI, 8501 para Streamlit)
EXPOSE 8000
EXPOSE 8501