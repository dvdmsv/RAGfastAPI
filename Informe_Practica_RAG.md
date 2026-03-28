# Sistema RAG Local Interactivo con Agente ReAct

---

**Alumno:** David

**Fecha:** Marzo 2026

**Asignatura:** Sistemas Aprendizaje Automatico + Modelos Inteligencia Artificial

**Centro:** IES Ribera de Castilla

---

## 1. Descripcion del sistema

El presente proyecto implementa un sistema RAG (Retrieval-Augmented Generation) 100% local y autonomo. Su objetivo es permitir a los usuarios consultar y extraer informacion de una base de conocimiento documental interna mediante lenguaje natural, garantizando la privacidad de los datos en todo momento, ya que no depende de APIs de pago ni envia datos a la nube.

El sistema se ha construido bajo una arquitectura de microservicios con un **Backend** (API REST en FastAPI) y un **Frontend** web interactivo (Streamlit), superando el enfoque de scripts por terminal propuesto en la practica base. Es capaz de:

- **Leer e indexar dinamicamente** documentos en formato PDF y TXT subidos por el usuario a traves de la interfaz web.
- **Responder preguntas en lenguaje natural** fundamentando sus respuestas exclusivamente en el contexto recuperado de los documentos indexados.
- **Prevenir alucinaciones**, indicando explicitamente cuando la informacion no se encuentra en el corpus documental.
- **Utilizar un agente ReAct** que decide autonomamente que herramienta (tool) usar segun la consulta del usuario, expandiendo sus capacidades mas alla de la simple recuperacion de texto.
- **Mantener conversacion general** (saludos, preguntas de contexto) sin necesidad de documentos indexados, gracias a un mecanismo de fallback a LLM directo.

El conjunto documental es completamente dinamico: el usuario sube los documentos que desee desde la interfaz web. No hay un corpus fijo predefinido, sino que el sistema acepta cualquier PDF o TXT proporcionado.

---

## 2. Arquitectura implementada

La arquitectura cumple integramente con las 5 capas requeridas en la especificacion, evolucionandolas hacia un entorno de microservicios orientado a produccion.

### Diagrama de arquitectura

```
+------------------+         HTTP/REST         +-------------------+
|                  |  -----------------------> |                   |
|    FRONTEND      |                           |     BACKEND       |
|   (Streamlit)    |  <----------------------- |    (FastAPI)       |
|   Puerto 8501    |    StreamingResponse      |   Puerto 8000     |
+------------------+                           +-------------------+
                                                        |
                                                        v
                                               +-------------------+
                                               |   AGENTE ReAct    |
                                               |   (LlamaIndex)    |
                                               +-------------------+
                                                   /    |    \
                                                  v     v     v
                                   +-----------+ +----+ +----------+
                                   | consultar | |hora| | buscar   |
                                   | documentos| |    | | palabra  |
                                   +-----------+ +----+ +----------+
                                         |
                                         v
                              +---------------------+
                              |   QDRANT (Hibrido)  |
                              | Denso + Sparse/BM25 |
                              |  ./data/qdrant_db/  |
                              +---------------------+
                                         ^
                                         |  Embeddings
                              +---------------------+
                              |    HuggingFace      |
                              | paraphrase-multi-   |
                              | lingual-MiniLM-L12  |
                              +---------------------+

                              +---------------------+
                              |     LM Studio       |
                              |   Puerto 1234       |
                              | (Inferencia local)  |
                              +---------------------+
```

### Descripcion de las 5 capas

| # | Capa | Implementacion |
|---|------|----------------|
| 1 | **Ingesta de documentos** | `SimpleDirectoryReader` (LlamaIndex) + `SentenceSplitter` con chunk_size=700 y overlap=120. Carga dinamica via interfaz web. |
| 2 | **Embeddings y Vectorstore** | `HuggingFaceEmbedding` (paraphrase-multilingual-MiniLM-L12-v2) + **Qdrant** con busqueda hibrida (densa + sparse/BM25), persistido en disco. |
| 3 | **Retriever** | `VectorStoreIndex` de LlamaIndex con `similarity_top_k=6` y postprocesador `SimilarityPostprocessor` (cutoff=0.30). |
| 4 | **Cadena RAG** | Conexion a LM Studio via interfaz compatible con OpenAI. System Prompt estricto que obliga al modelo a citar fuentes y rechazar respuestas sin evidencia. Streaming de tokens. |
| 5 | **Agente + Tools** | `ReActAgent` (LlamaIndex) con 3 herramientas: `consultar_documentos`, `obtener_fecha_hora` y `buscar_palabra_clave`. |

### Agente ReAct y sus herramientas

El nucleo del sistema es un **ReActAgent** que implementa el paradigma *Reasoning and Acting*. El agente razona sobre la pregunta del usuario y decide autonomamente que herramienta invocar:

1. **`consultar_documentos`** -- Realiza una consulta semantica contra la base vectorial Qdrant. Devuelve la respuesta del LLM junto con las fuentes consultadas (archivo, pagina, puntuacion de relevancia).
2. **`obtener_fecha_hora`** -- Devuelve la fecha, hora y dia de la semana actuales del sistema. Dota al agente de conciencia temporal, algo de lo que los LLMs carecen por defecto.
3. **`buscar_palabra_clave`** -- Busca una palabra clave exacta en todos los documentos indexados y devuelve los fragmentos con contexto circundante (150 caracteres antes y despues).

### Despliegue contenerizado (mejora sobre los requisitos base)

Como valor anadido, el sistema completo ha sido contenerizado con Docker y Docker Compose, ofreciendo **3 variantes de despliegue**:

| Variante | Caso de uso | Comando |
|----------|-------------|---------|
| A: Docker estandar | Docker nativo (Linux/macOS/Windows) | `docker compose up --build` |
| B: Docker + WSL | Docker en WSL2 + LM Studio en Windows | `./scripts/docker-compose-wsl.sh up --build` |
| C: Nativo (sin Docker) | Ejecucion directa de Python | `bash start_local.sh` |

---

## 3. Justificacion del modelo y parametros

### Eleccion del framework: LlamaIndex en lugar de LangChain

Aunque la practica propone LangChain como orquestador, se ha tomado la decision de ingenieria de utilizar **LlamaIndex**. La justificacion es la siguiente:

- **LlamaIndex esta concebido especificamente para RAG**: mientras que LangChain es un framework generalista, LlamaIndex se especializa en ingesta de datos, indexacion y recuperacion documental.
- **Soporte nativo de busqueda hibrida**: LlamaIndex integra de forma nativa la combinacion de busqueda densa (embeddings) con busqueda sparse (BM25) a traves de Qdrant, sin necesidad de wrappers adicionales.
- **Agente ReAct nativo**: LlamaIndex ofrece `ReActAgent` como componente de primera clase, con soporte para streaming de eventos (`AgentStream`).

### Eleccion de Qdrant en lugar de ChromaDB

Se sustituyo ChromaDB por **Qdrant** por razones tecnicas:

- Qdrant esta escrito en Rust, optimizado para entornos contenerizados.
- ChromaDB utiliza SQLite internamente, lo que puede generar bloqueos de base de datos en Docker con multiples accesos concurrentes.
- Qdrant soporta busqueda hibrida (densa + sparse/BM25) de forma nativa, ideal para encontrar codigos de referencia y nombres propios exactos.

### Eleccion del modelo LLM

El sistema es **agnostico al modelo**: cualquier modelo compatible con la API de OpenAI servido por LM Studio puede utilizarse. Se recomienda `qwen/qwen3.5-9b` por su equilibrio entre consumo de hardware y capacidad de seguimiento de instrucciones. La eleccion de LM Studio garantiza inferencia 100% local, sin costes de API ni fuga de datos.

Se implemento un **monkey-patching** en `rag.py` para compatibilidad con modelos custom de LM Studio, ya que LlamaIndex internamente valida nombres de modelo contra una lista fija de modelos de OpenAI.

### Parametros de ingesta

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `chunk_size` | 700 | Equilibrio entre preservar contexto suficiente para que el LLM elabore respuestas coherentes y mantener la densidad semantica del vector. Fragmentos mas grandes (>1000) diluyen la semantica; mas pequenos (<300) pierden contexto. |
| `chunk_overlap` | 120 | Evita la fragmentacion de ideas en los limites de corte. Actua como "adhesivo" contextual, garantizando que conceptos clave no se pierdan al caer entre dos fragmentos. |

### Parametros de recuperacion

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `similarity_top_k` | 6 | Proporciona contexto suficiente para cruzar datos de multiples fragmentos sin desbordar la ventana de contexto del modelo ni ralentizar excesivamente la inferencia local. |
| `similarity_cutoff` | 0.30 | Umbral de relevancia minima. Filtra fragmentos poco relevantes que introducirian ruido, pero es suficientemente permisivo para no dejar respuestas vacias. |
| `source_top_k` | 5 | Limita las fuentes citadas al usuario a las 5 mas relevantes. |

### Parametros de generacion

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `temperature` | 0.1 | Al tratarse de un asistente RAG documental, se prioriza la veracidad y el determinismo. Una temperatura baja fuerza al modelo a seleccionar los tokens mas probables, eliminando la "creatividad" y evitando que invente informacion. |

---

## 4. Resultados

Se presentan 5 ejemplos de interaccion que demuestran el funcionamiento de las distintas herramientas del agente.

### Ejemplo 1: Consulta temporal

| | |
|---|---|
| **Pregunta** | "Que hora es?" |
| **Tool utilizada** | `obtener_fecha_hora` |
| **Respuesta esperada** | El agente reconoce que es una consulta temporal, invoca la herramienta de fecha/hora y devuelve la fecha, hora y dia de la semana actuales del sistema. |
| **Evaluacion** | Correcta. El agente decide autonomamente usar la herramienta adecuada sin intentar buscar en los documentos. Demuestra la capacidad del paradigma ReAct para enrutar consultas. |

### Ejemplo 2: Consulta documental

| | |
|---|---|
| **Pregunta** | "De que trata el documento?" |
| **Tool utilizada** | `consultar_documentos` |
| **Respuesta esperada** | El agente realiza una busqueda semantica en Qdrant, recupera los fragmentos mas relevantes y genera un resumen del contenido, citando el nombre del archivo fuente, la pagina y la puntuacion de relevancia. |
| **Evaluacion** | Correcta. Las respuestas se fundamentan en el contexto recuperado y se incluyen las fuentes citadas. La calidad depende del modelo LLM cargado en LM Studio. |

### Ejemplo 3: Busqueda de palabra clave

| | |
|---|---|
| **Pregunta** | "Busca la palabra 'Python' en los documentos" |
| **Tool utilizada** | `buscar_palabra_clave` |
| **Respuesta esperada** | El agente realiza una busqueda literal (no semantica) en todos los documentos indexados, devolviendo los fragmentos exactos donde aparece la palabra junto con contexto circundante. |
| **Evaluacion** | Correcta. Complementa la busqueda semantica del RAG con una busqueda exacta, util para nombres propios, codigos de referencia o terminos tecnicos especificos. |

### Ejemplo 4: Conversacion general (sin herramientas)

| | |
|---|---|
| **Pregunta** | "Hola" |
| **Tool utilizada** | Ninguna (respuesta directa del agente) |
| **Respuesta esperada** | El agente responde de forma amable sin necesidad de usar ninguna herramienta, saludando al usuario y mencionandole que puede hacer preguntas sobre los documentos indexados. |
| **Evaluacion** | Correcta. Si el agente devuelve una respuesta vacia, se activa un mecanismo de fallback que consulta directamente al LLM con un prompt conversacional. |

### Ejemplo 5: Informacion ausente en los documentos

| | |
|---|---|
| **Pregunta** | "Cual es la capital de Marte?" |
| **Tool utilizada** | `consultar_documentos` |
| **Respuesta esperada** | El agente busca en los documentos, no encuentra informacion relevante (el cutoff de similitud filtra los fragmentos) y responde explicitamente que no tiene informacion sobre eso en los documentos indexados. |
| **Evaluacion** | Correcta. El system prompt instruye al modelo para que indique claramente cuando la evidencia no es suficiente, previniendo alucinaciones. |

---

## 5. Limitaciones y mejoras

### Limitaciones actuales

- **Dependencia del formato ReAct en el modelo**: El correcto funcionamiento del agente depende de la capacidad del modelo LLM para seguir el formato Thought/Action/Observation del paradigma ReAct. Modelos mas pequenos o menos capaces pueden fallar al generar el formato esperado, provocando respuestas vacias o bucles.
- **Ausencia de memoria conversacional entre turnos del agente**: Cada invocacion del agente es independiente. El agente no recuerda lo que se dijo en turnos anteriores dentro de la misma sesion, limitando las preguntas de seguimiento (e.g., "Dime mas sobre ese ultimo punto").
- **Streaming diferido**: Aunque el sistema implementa streaming, el agente ReAct debe completar todo su ciclo de razonamiento (Thought -> Action -> Observation -> Answer) antes de emitir la respuesta final. El usuario no ve tokens en tiempo real durante la fase de razonamiento.
- **Cuellos de botella en hardware**: La velocidad de inferencia esta ligada a la CPU/GPU local. En escenarios de multiples peticiones concurrentes, el modelo encola consultas, elevando los tiempos de respuesta.
- **Monkey-patching para compatibilidad**: La integracion con LM Studio requiere parches en tiempo de ejecucion sobre las utilidades internas de LlamaIndex para aceptar nombres de modelo custom, lo que puede romperse con actualizaciones del framework.

### Mejoras futuras

- **Memoria conversacional**: Integrar un historial de mensajes en el agente para permitir preguntas de seguimiento contextualizadas, similar a `ConversationBufferMemory`.
- **Streaming token a token real**: Implementar streaming durante la fase de razonamiento del agente, no solo al final, para mejorar la experiencia de usuario.
- **FunctionAgent como alternativa**: Anadir un `FunctionAgent` como alternativa para modelos con soporte nativo de tool calling (function calling), evitando la dependencia del formato ReAct textual.
- **Herramientas adicionales**: Ampliar el conjunto de tools del agente con capacidades como busqueda web, calculadora, o resumen automatico de documentos.
- **Tests automatizados**: El proyecto carece de tests. Anadir tests unitarios para las funciones de RAG y tests de integracion para los endpoints de la API mejoraria la mantenibilidad.
- **Autenticacion y seguridad**: Anadir autenticacion al backend y restringir CORS para un posible despliegue fuera del entorno local.
