import asyncio
import concurrent.futures
import os
import re
from datetime import datetime
from typing import Any, Generator

from dotenv import load_dotenv
import requests

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import AgentStream
import tiktoken
import llama_index.llms.openai.utils as _openai_utils

# ==========================================
# MONKEY-PATCHES: Permitir modelos custom de LM Studio
# ==========================================
_original_ctx = _openai_utils.openai_modelname_to_contextsize
def _patched_ctx(modelname: str) -> int:
    try:
        return _original_ctx(modelname)
    except ValueError:
        return 4096
_openai_utils.openai_modelname_to_contextsize = _patched_ctx

_original_is_chat = _openai_utils.is_chat_model
def _patched_is_chat(model: str) -> bool:
    if model not in _openai_utils.ALL_AVAILABLE_MODELS:
        return True
    return _original_is_chat(model)
_openai_utils.is_chat_model = _patched_is_chat

from llama_index.llms.openai import OpenAI
import llama_index.llms.openai.base as _openai_base
import llama_index.llms.openai.responses as _openai_responses
_openai_base.openai_modelname_to_contextsize = _patched_ctx
_openai_base.is_chat_model = _patched_is_chat
_openai_responses.openai_modelname_to_contextsize = _patched_ctx

_fallback_enc = tiktoken.get_encoding("cl100k_base")
_original_enc = tiktoken.encoding_for_model
def _patched_enc(model_name: str):
    try:
        return _original_enc(model_name)
    except KeyError:
        return _fallback_enc
tiktoken.encoding_for_model = _patched_enc
_openai_base.tiktoken = tiktoken

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

load_dotenv()


def normalizar_openai_api_base(raw_url: str) -> str:
    url = raw_url.rstrip("/")
    if not re.search(r"/v\d+$", url):
        url = f"{url}/v1"
    return url


openai_api_base = normalizar_openai_api_base(
    os.getenv("OPENAI_API_BASE", "http://host.docker.internal:1234/v1")
)
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "700"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
DEFAULT_SIMILARITY_TOP_K = int(os.getenv("RAG_SIMILARITY_TOP_K", "6"))
DEFAULT_SIMILARITY_CUTOFF = float(os.getenv("RAG_SIMILARITY_CUTOFF", "0.30"))
DEFAULT_SOURCE_TOP_K = int(os.getenv("RAG_SOURCE_TOP_K", "5"))
DEFAULT_USE_RERANKER = os.getenv("RAG_USE_RERANKER", "false").lower() == "true"
DEFAULT_RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "3"))
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO DE LENGUAJE
# ==========================================
llm = OpenAI(
    api_base=openai_api_base,
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    model=DEFAULT_LLM_MODEL,
    temperature=0.1
)

# ==========================================
# 2. CONFIGURACIÓN DE EMBEDDINGS LOCALES
# ==========================================
embed_model = HuggingFaceEmbedding(
    model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = SentenceSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
)

# ==========================================
# 3. BASE DE DATOS VECTORIAL (HÍBRIDA)
# ==========================================
qdrant_path = os.getenv("QDRANT_PATH", "./data/qdrant_db")
client = qdrant_client.QdrantClient(path=qdrant_path)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="documentos_rag",
    enable_hybrid=True
)

# ==========================================
# 4. MEMORIA MULTI-USUARIO (Agentes por sesión)
# ==========================================
agentes_por_sesion = {}


def listar_modelos_lm_studio() -> dict[str, Any]:
    base_url = openai_api_base.rstrip("/")
    url_modelos = f"{base_url}/models"
    try:
        respuesta = requests.get(
            url_modelos,
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'lm-studio')}"},
            timeout=10,
        )
        respuesta.raise_for_status()
        payload = respuesta.json()
        modelos = [item.get("id") for item in payload.get("data", []) if item.get("id")]
        return {"modelos": modelos, "modelo_actual": DEFAULT_LLM_MODEL}
    except requests.RequestException as exc:
        return {
            "modelos": [],
            "modelo_actual": DEFAULT_LLM_MODEL,
            "error": f"No se pudieron obtener los modelos de LM Studio: {exc}",
        }


def construir_configuracion_rag(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    similarity_top_k: int | None = None,
    similarity_cutoff: float | None = None,
    source_top_k: int | None = None,
    usar_reranker: bool | None = None,
    rerank_top_n: int | None = None,
) -> dict[str, Any]:
    config = {
        "chunk_size": chunk_size or DEFAULT_CHUNK_SIZE,
        "chunk_overlap": chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP,
        "similarity_top_k": similarity_top_k or DEFAULT_SIMILARITY_TOP_K,
        "similarity_cutoff": similarity_cutoff if similarity_cutoff is not None else DEFAULT_SIMILARITY_CUTOFF,
        "source_top_k": source_top_k or DEFAULT_SOURCE_TOP_K,
        "usar_reranker": usar_reranker if usar_reranker is not None else DEFAULT_USE_RERANKER,
        "rerank_top_n": rerank_top_n or DEFAULT_RERANK_TOP_N,
    }
    config["chunk_size"] = max(200, config["chunk_size"])
    config["chunk_overlap"] = max(0, min(config["chunk_overlap"], config["chunk_size"] // 2))
    config["similarity_top_k"] = max(1, config["similarity_top_k"])
    config["similarity_cutoff"] = max(0.0, min(config["similarity_cutoff"], 1.0))
    config["source_top_k"] = max(1, min(config["source_top_k"], config["similarity_top_k"]))
    config["rerank_top_n"] = max(1, min(config["rerank_top_n"], config["similarity_top_k"]))
    return config


def crear_text_splitter(configuracion_rag: dict[str, Any]) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=configuracion_rag["chunk_size"],
        chunk_overlap=configuracion_rag["chunk_overlap"],
    )


def crear_postprocesadores(configuracion_rag: dict[str, Any]) -> list[Any]:
    postprocesadores: list[Any] = [
        SimilarityPostprocessor(similarity_cutoff=configuracion_rag["similarity_cutoff"])
    ]
    if not configuracion_rag["usar_reranker"]:
        return postprocesadores
    try:
        try:
            from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
        except ImportError:
            from llama_index.core.postprocessor import SentenceTransformerRerank
        postprocesadores.append(
            SentenceTransformerRerank(
                top_n=configuracion_rag["rerank_top_n"],
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
        )
    except ImportError:
        pass
    return postprocesadores


def crear_clave_configuracion(configuracion_rag: dict[str, Any]) -> tuple[Any, ...]:
    return (
        configuracion_rag["similarity_top_k"],
        configuracion_rag["similarity_cutoff"],
        configuracion_rag["source_top_k"],
        configuracion_rag["usar_reranker"],
        configuracion_rag["rerank_top_n"],
    )


def es_respuesta_vacia(texto: str) -> bool:
    sin_think = re.sub(r"<think>[\s\S]*?</think>", "", texto).strip()
    sin_think = re.sub(r"<think>[\s\S]*", "", sin_think).strip()
    return not sin_think or sin_think == "Empty Response"


def construir_system_prompt(prompt_personalizado: str | None = None) -> str:
    if prompt_personalizado:
        return prompt_personalizado
    return (
        "Eres un asistente experto en analisis documental. Responde siempre en espanol. "
        "Dispones de herramientas (tools) para consultar documentos indexados, obtener la fecha y hora, "
        "y buscar palabras clave. Decide que herramienta usar segun la pregunta del usuario. "
        "Para preguntas sobre documentos, usa la herramienta de consulta documental y basa tu respuesta "
        "UNICAMENTE en la informacion recuperada. No inventes datos. "
        "Cuando respondas con informacion factual de los documentos, cita el nombre del archivo fuente. "
        "Si la evidencia no es suficiente, indicalo claramente. "
        "Para saludos o conversacion general, responde de forma amable sin necesidad de usar herramientas. "
        "IMPORTANTE: NO uses etiquetas <think> ni </think>. Usa UNICAMENTE el formato Thought/Action/Answer "
        "descrito en las instrucciones del sistema."
    )


# ==========================================
# 5. TOOLS DEL AGENTE
# ==========================================
def tool_consultar_documentos(pregunta: str) -> str:
    """Consulta la base de conocimiento interna para responder preguntas
    sobre los documentos indexados. Usa esta herramienta cuando el usuario
    pregunte sobre el contenido de los documentos, quiera resumenes,
    o necesite informacion especifica de los archivos subidos."""
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        query_engine = index.as_query_engine(
            similarity_top_k=DEFAULT_SIMILARITY_TOP_K,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=DEFAULT_SIMILARITY_CUTOFF)
            ],
        )
        respuesta = query_engine.query(pregunta)

        # Extraer fuentes citadas
        fuentes = []
        if respuesta.source_nodes:
            for nodo in respuesta.source_nodes[:DEFAULT_SOURCE_TOP_K]:
                archivo = nodo.node.metadata.get("file_name", "Desconocido")
                pagina = nodo.node.metadata.get("page_label", "N/A")
                score = getattr(nodo, "score", 0.0)
                fuentes.append(f"- {archivo} (Pag: {pagina}, Relevancia: {score:.2f})")

        texto_respuesta = str(respuesta)
        if fuentes:
            texto_respuesta += "\n\nFuentes consultadas:\n" + "\n".join(fuentes)
        return texto_respuesta
    except Exception as e:
        return f"Error al consultar documentos: {str(e)}"


def tool_obtener_fecha_hora(consulta: str = "") -> str:
    """Devuelve la fecha y hora actual del sistema. Usa esta herramienta
    cuando el usuario pregunte por la hora, la fecha, que dia es hoy,
    o cualquier informacion temporal."""
    ahora = datetime.now()
    return (
        f"Fecha actual: {ahora.strftime('%d/%m/%Y')}\n"
        f"Hora actual: {ahora.strftime('%H:%M:%S')}\n"
        f"Dia de la semana: {ahora.strftime('%A')}"
    )


def tool_buscar_palabra_clave(palabra_clave: str) -> str:
    """Busca una palabra clave exacta en todos los documentos indexados.
    Usa esta herramienta cuando el usuario quiera encontrar menciones
    especificas de un termino, codigo, nombre propio o referencia exacta
    en los documentos. Devuelve los fragmentos que contienen la palabra."""
    try:
        directorio_docs = "./data/docs"
        if not os.path.exists(directorio_docs):
            return "No hay documentos indexados."

        resultados = []
        documentos = SimpleDirectoryReader(directorio_docs).load_data()
        palabra_lower = palabra_clave.lower()

        for doc in documentos:
            texto = doc.text
            if palabra_lower in texto.lower():
                archivo = doc.metadata.get("file_name", "Desconocido")
                # Extraer contexto alrededor de la palabra clave
                idx = texto.lower().find(palabra_lower)
                inicio = max(0, idx - 150)
                fin = min(len(texto), idx + len(palabra_clave) + 150)
                fragmento = texto[inicio:fin].strip()
                resultados.append(f"En '{archivo}': ...{fragmento}...")

        if not resultados:
            return f"No se encontro '{palabra_clave}' en ningun documento indexado."

        return f"Se encontro '{palabra_clave}' en {len(resultados)} fragmento(s):\n\n" + "\n\n".join(resultados[:5])
    except Exception as e:
        return f"Error en la busqueda: {str(e)}"


# ==========================================
# 6. CREACIÓN DEL AGENTE
# ==========================================
def crear_agente(custom_llm, system_prompt: str) -> ReActAgent:
    """Crea un agente ReAct con las tools disponibles."""
    tools = [
        FunctionTool.from_defaults(
            fn=tool_consultar_documentos,
            name="consultar_documentos",
            description=(
                "Consulta la base de conocimiento interna para responder preguntas "
                "sobre los documentos indexados. Usala para cualquier pregunta sobre "
                "el contenido de los archivos subidos."
            ),
        ),
        FunctionTool.from_defaults(
            fn=tool_obtener_fecha_hora,
            name="obtener_fecha_hora",
            description=(
                "Devuelve la fecha y hora actual del sistema. Usala cuando el usuario "
                "pregunte por la hora, la fecha o que dia es hoy."
            ),
        ),
        FunctionTool.from_defaults(
            fn=tool_buscar_palabra_clave,
            name="buscar_palabra_clave",
            description=(
                "Busca una palabra clave exacta en todos los documentos indexados. "
                "Usala para encontrar menciones especificas de un termino, codigo, "
                "nombre propio o referencia exacta."
            ),
        ),
    ]

    return ReActAgent(
        tools=tools,
        llm=custom_llm,
        system_prompt=system_prompt,
        verbose=True,
        early_stopping_method="generate",
    )


# ==========================================
# 7. INDEXACIÓN DE DOCUMENTOS
# ==========================================
def inicializar_y_cargar_datos(
    directorio_docs: str = "./data/docs",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> str:
    if not os.path.exists(directorio_docs) or not os.listdir(directorio_docs):
        return "El directorio esta vacio o no existe."

    configuracion_rag = construir_configuracion_rag(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documentos = SimpleDirectoryReader(directorio_docs).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    parser = crear_text_splitter(configuracion_rag)
    nodos = parser.get_nodes_from_documents(documentos)

    VectorStoreIndex(nodes=nodos, storage_context=storage_context)

    global agentes_por_sesion
    agentes_por_sesion.clear()

    return (
        f"Se han indexado {len(documentos)} documentos en {len(nodos)} fragmentos "
        f"(chunk_size={configuracion_rag['chunk_size']}, overlap={configuracion_rag['chunk_overlap']})."
    )


# ==========================================
# 8. RESPUESTA CON AGENTE (Streaming)
# ==========================================
def obtener_respuesta_rag_stream(
    pregunta: str,
    session_id: str = "usuario_default",
    system_prompt: str | None = None,
    temperature: float = 0.1,
    similarity_top_k: int | None = None,
    similarity_cutoff: float | None = None,
    source_top_k: int | None = None,
    usar_reranker: bool | None = None,
    rerank_top_n: int | None = None,
    llm_model: str | None = None,
) -> Generator[str, None, None]:
    global agentes_por_sesion
    contenido_generado = []

    try:
        system_prompt = construir_system_prompt(system_prompt)
        modelo_a_usar = llm_model or DEFAULT_LLM_MODEL

        # Actualizar los defaults de las tools según la configuración del usuario
        global DEFAULT_SIMILARITY_TOP_K, DEFAULT_SIMILARITY_CUTOFF, DEFAULT_SOURCE_TOP_K
        if similarity_top_k:
            DEFAULT_SIMILARITY_TOP_K = similarity_top_k
        if similarity_cutoff is not None:
            DEFAULT_SIMILARITY_CUTOFF = similarity_cutoff
        if source_top_k:
            DEFAULT_SOURCE_TOP_K = source_top_k

        # Decidir si recrear el agente
        recreate = False
        agent_data = agentes_por_sesion.get(session_id)
        if not agent_data:
            recreate = True
        else:
            if (
                agent_data.get("system_prompt") != system_prompt
                or agent_data.get("temperature") != temperature
                or agent_data.get("llm_model") != modelo_a_usar
            ):
                recreate = True

        if recreate:
            custom_llm = OpenAI(
                api_base=openai_api_base,
                api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
                model=modelo_a_usar,
                temperature=temperature
            )

            agente = crear_agente(custom_llm, system_prompt)

            agentes_por_sesion[session_id] = {
                "agent": agente,
                "llm": custom_llm,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "llm_model": modelo_a_usar,
            }

        agente = agentes_por_sesion[session_id]["agent"]

        # Ejecutar el agente con el workflow async y recoger deltas
        async def _run_agent():
            """Ejecuta el agente ReAct y recopila los deltas de streaming."""
            deltas = []
            handler = agente.run(user_msg=pregunta, max_iterations=10)
            async for event in handler.stream_events():
                if isinstance(event, AgentStream) and event.delta:
                    deltas.append(event.delta)
            # Obtener resultado final
            final = await handler
            # Extraer texto de la respuesta final del agente
            respuesta_final = ""
            if hasattr(final, 'result'):
                result_obj = final.result
                if hasattr(result_obj, 'response') and hasattr(result_obj.response, 'content'):
                    respuesta_final = result_obj.response.content or ""
                else:
                    respuesta_final = str(result_obj)
            elif hasattr(final, 'response'):
                respuesta_final = str(final.response)
            else:
                respuesta_final = str(final)
            return deltas, respuesta_final

        # Ejecutar async desde contexto sync (FastAPI corre en thread pool para endpoints sync)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            deltas, respuesta_final = pool.submit(
                lambda: asyncio.run(_run_agent())
            ).result()

        # Usar deltas de streaming si los hubo, sino la respuesta final
        texto_completo = "".join(deltas) if deltas else respuesta_final

        # Streaming con procesamiento de think tags en tiempo real
        buffer = texto_completo
        in_think = False
        pos = 0

        while pos < len(buffer):
            if not in_think:
                think_start = buffer.find("<think>", pos)
                if think_start != -1:
                    # Emitir contenido antes del think tag
                    pre = buffer[pos:think_start]
                    if pre:
                        contenido_generado.append(pre)
                        yield pre
                    yield "\n\n> 💭 **Razonamiento de la IA:**\n> "
                    pos = think_start + len("<think>")
                    in_think = True
                else:
                    # No hay más think tags, emitir el resto
                    rest = buffer[pos:]
                    if rest:
                        contenido_generado.append(rest)
                        yield rest
                    pos = len(buffer)
            else:
                think_end = buffer.find("</think>", pos)
                if think_end != -1:
                    think_content = buffer[pos:think_end]
                    yield think_content.replace("\n", "\n> ")
                    yield "\n\n---\n\n"
                    pos = think_end + len("</think>")
                    in_think = False
                else:
                    # Think tag sin cerrar, emitir lo que queda
                    rest = buffer[pos:]
                    yield rest.replace("\n", "\n> ")
                    pos = len(buffer)

        # Si el agente devolvió respuesta vacía, fallback a LLM directo
        respuesta_texto = "".join(contenido_generado).strip()
        if es_respuesta_vacia(respuesta_texto):
            from llama_index.core.llms import ChatMessage, MessageRole
            custom_llm = agentes_por_sesion[session_id].get("llm", llm)
            prompt_libre = (
                "Eres un asistente conversacional amable. Responde siempre en espanol. "
                "Responde de forma natural y breve. Si el usuario te saluda, saluda de vuelta "
                "y mencionale que puede hacerte preguntas sobre los documentos que tiene indexados."
            )
            mensajes = [
                ChatMessage(role=MessageRole.SYSTEM, content=prompt_libre),
                ChatMessage(role=MessageRole.USER, content=pregunta),
            ]
            for chunk in custom_llm.stream_chat(mensajes):
                yield chunk.delta

    except Exception as e:
        yield f"Error al consultar el modelo: {str(e)}"
