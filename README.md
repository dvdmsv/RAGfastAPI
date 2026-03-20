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
* **Inferencia LLM:** LM Studio (ejecución 100% local, sin APIs de terceros ni fuga de datos). Soporta **Modelos de Razonamiento**, ya que el backend intercepta y formatea correctamente las etiquetas `<think>` emitidas por modelos como DeepSeek R1 en el UI.
* **Despliegue:** Docker y Docker Compose para garantizar reproducibilidad total, o despliegue local nativo.
## 🚀 1. Configuración de LM Studio (Requisito Previo)

El sistema RAG es agnóstico al modelo gracias a la API compatible con OpenAI de LM Studio. 

1. Descarga e instala [LM Studio](https://lmstudio.ai).
2. Descarga cualquier modelo compatible con tu hardware.
3. Ve a la pestaña **"Local Server"**.
4. **⚠️ PASO CRÍTICO PARA DOCKER:** En la configuración de red del servidor (Network), activa la opción **"Serve on local network"** (CORS). Si LM Studio solo escucha en `127.0.0.1`, los contenedores no podrán alcanzarlo salvo en escenarios muy concretos como ciertos reenvíos de WSL.
5. Inicia el servidor. Se ejecutará por defecto en el puerto `1234`.

## 🛠️ 2. Instalación y Orden de Ejecución

Según la rúbrica, las dependencias están congeladas en el archivo `requirements.txt`. Sin embargo, gracias a la arquitectura basada en Docker, **no es necesario instalar dependencias en tu entorno local ni ejecutar scripts por separado**. El orquestador se encarga de todo.

1. Abre una terminal en la raíz del proyecto.
2. Define la URL de LM Studio en el archivo `.venv` según tu entorno.

### Cuadro de decisión rápido

| Caso | Compose a usar | Cómo arrancar |
|---|---|---|
| Docker corre normalmente en tu equipo y LM Studio también (Variante A) | `docker-compose.yml` | `docker compose up --build` |
| Docker corre dentro de WSL y LM Studio corre en Windows (Variante B) | `docker-compose.wsl.yml` | `./scripts/docker-compose-wsl.sh up --build` |
| No quieres usar Docker; ejecución directa de Python (Variante C) | *No aplica* | `bash start_local.sh` |

Regla práctica:

- si LM Studio debe alcanzarse como `http://host.docker.internal:1234/v1`, usa Variante A.
- si LM Studio debe alcanzarse como `http://localhost:1234/v1` desde WSL, usa Variante B.
- si prefieres ejecutar el código de Python directamente de fondo en tu máquina, usa Variante C.

### Variante A: Docker "normal" en el mismo equipo (recomendada por defecto)

Usa esta variante si:

- Docker corre directamente en Linux, Windows o macOS.
- LM Studio corre en la misma máquina que Docker.
- `host.docker.internal` funciona en tu instalación de Docker.

Configuración recomendada en `.venv`:
   ```dotenv
   OPENAI_API_BASE=http://host.docker.internal:1234/v1
   OPENAI_API_KEY=lm-studio
   LLM_MODEL=gpt-3.5-turbo
   ```

Arranque:
   ```bash
   docker compose up --build
   ```

Esta variante usa el archivo `docker-compose.yml` y mantiene la red normal de Docker:

- backend en `http://localhost:8000`
- frontend en `http://localhost:8501`
- LM Studio accesible desde el contenedor como `http://host.docker.internal:1234/v1`

### Variante B: Docker en WSL + LM Studio en Windows

Usa esta variante si:

- Docker se ejecuta dentro de WSL.
- LM Studio se ejecuta en Windows.
- Desde WSL puedes hacer `curl http://localhost:1234/v1/models` pero Docker no resuelve bien `host.docker.internal`.

Configuración recomendada en `.venv`:
   ```dotenv
   OPENAI_API_BASE=http://localhost:1234/v1
   OPENAI_API_KEY=lm-studio
   LLM_MODEL=gpt-3.5-turbo
   ```

Arranque:
   ```bash
   ./scripts/docker-compose-wsl.sh up --build
   ```

Esta variante usa el archivo `docker-compose.wsl.yml` y cambia la red a `host` para que los contenedores usen el mismo `localhost` de WSL:

- backend en `http://localhost:8000`
- frontend en `http://localhost:8501`
- LM Studio accesible desde el contenedor como `http://localhost:1234/v1`

### Variante C: Ejecución Local Nativa (Sin Docker)

Usa esta variante si:

- Tienes problemas de red con Docker o Windows Subsystem for Linux.
- Prefieres no usar contenedores y apoyarte directamente en el entorno virtual de tu SO.

Arranque automático:
   ```bash
   bash start_local.sh
   ```

Este script automatizado buscará y creará automáticamente el entorno virtual, instalará las dependencias de `requirements.txt` y lanzará los servicios de FastAPI y Streamlit sin bloquear el terminal.

## 📚 3. Diferencia entre `host.docker.internal` y `network_mode: host`

- `host.docker.internal`: nombre DNS especial para llegar al host desde una red normal de Docker.
- `network_mode: host`: el contenedor comparte directamente la red del host y usa su mismo `localhost`.

Regla práctica:

- usa `host.docker.internal` cuando quieras la opción más portable
- usa `network_mode: host` cuando estés en WSL y `localhost:1234` ya funcione desde Linux pero Docker no llegue bien al host

## ✅ 4. Verificaciones rápidas

Antes de levantar los contenedores, puedes comprobar cuál variante te conviene.

### Si vas a usar la variante A

Prueba desde un contenedor temporal:
```bash
docker run --rm --add-host host.docker.internal:host-gateway alpine sh -c "apk add --no-cache curl >/dev/null && curl http://host.docker.internal:1234/v1/models"
```

### Si vas a usar la variante B

Prueba desde WSL:
```bash
curl http://localhost:1234/v1/models
```

## 🧠 5. Uso del sistema RAG

Una vez levantado el sistema:

1. Abre el frontend en `http://localhost:8501`.
2. Sube al menos un documento PDF/TXT.
3. Si prefieres indexar manualmente, puedes llamar al endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/indexar
   ```
4. Si intentas chatear sin haber indexado documentos, el backend no tendrá creada la colección `documentos_rag`.

## 🧰 6. Comandos más comunes

### Variante A: entorno normal

Levantar:
```bash
docker compose up --build
```

Levantar en segundo plano:
```bash
docker compose up --build -d
```

Parar y eliminar contenedores:
```bash
docker compose down
```

Reconstruir desde cero:
```bash
docker compose build --no-cache
docker compose up
```

Ver logs:
```bash
docker compose logs -f
```

### Variante B: WSL + LM Studio en Windows

Levantar:
```bash
./scripts/docker-compose-wsl.sh up --build
```

Levantar en segundo plano:
```bash
./scripts/docker-compose-wsl.sh up --build -d
```

Parar y eliminar contenedores:
```bash
./scripts/docker-compose-wsl.sh down
```

Reconstruir desde cero:
```bash
docker compose -f docker-compose.wsl.yml build --no-cache
./scripts/docker-compose-wsl.sh up
```

Ver logs:
```bash
./scripts/docker-compose-wsl.sh logs -f
```

### Variante C: Ejecución Local Nativa (Sin Docker)

Levantar todo en segundo plano:
```bash
bash start_local.sh
```

Detener los servicios locales:
```bash
pkill -f uvicorn && pkill -f streamlit
```

Ver logs:
```bash
tail -f backend_local.log
tail -f frontend_local.log
```
