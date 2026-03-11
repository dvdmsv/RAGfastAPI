# 🧠 Sistema RAG Local Interactivo (FastAPI + Streamlit + Docker)

[cite_start]Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) 100% local [cite: 8][cite_start], desarrollado como Práctica Final para la asignatura de Sistemas Aprendizaje Automático + Modelos Inteligencia Artificial[cite: 1]. 

Para este despliegue, se ha diseñado una **arquitectura de microservicios contenerizada** que supera los requisitos base, ofreciendo una interfaz gráfica completa y un backend robusto.

## 🏗️ Arquitectura del Sistema
A diferencia de un enfoque basado puramente en scripts de terminal, este sistema se divide en:
* **Backend (Cerebro):** API REST construida con FastAPI.
* **Orquestación RAG:** LlamaIndex (optimizado para RAG documental puro y recuperación de metadatos/citas).
* **Vectorstore:** Qdrant (persistido localmente, con capacidades de búsqueda avanzada).
* **Frontend (Interfaz):** Streamlit, que unifica la ingesta de documentos y el agente de chat en una sola aplicación web.
* **Embeddings:** HuggingFace (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
* **Inferencia LLM:** LM Studio (ejecución 100% local, sin APIs de terceros ni fuga de datos).
* **Despliegue:** Docker y Docker Compose para garantizar reproducibilidad total.

## 🚀 1. Configuración de LM Studio (Requisito Previo)

El sistema RAG es agnóstico al modelo gracias a la API compatible con OpenAI de LM Studio. 

1. Descarga e instala [LM Studio](https://lmstudio.ai).
2. Descarga cualquier modelo compatible con tu hardware.
3. Ve a la pestaña **"Local Server"**.
4. **⚠️ PASO CRÍTICO PARA DOCKER:** En la configuración de red del servidor (Network), activa la opción **"Serve on local network"** (CORS) o asegúrate de que expone en `0.0.0.0` en lugar de `localhost`. Esto permite que el contenedor Docker se comunique con el anfitrión.
5. Inicia el servidor. Se ejecutará por defecto en el puerto `1234`.

## 🛠️ 2. Instalación y Orden de Ejecución

Según la rúbrica, las dependencias están congeladas en el archivo `requirements.txt`. Sin embargo, gracias a la arquitectura basada en Docker, **no es necesario instalar dependencias en tu entorno local ni ejecutar scripts por separado**. El orquestador se encarga de todo.

1. Abre una terminal en la raíz del proyecto.
2. Construye y levanta el entorno ejecutando:
   ```bash
   docker compose up --build