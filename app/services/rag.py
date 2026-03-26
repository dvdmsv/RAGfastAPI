import os
import re
from datetime import datetime
from typing import Any, Generator

from dotenv import load_dotenv
import requests

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
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

# Activamos la búsqueda híbrida (Dense + Sparse/BM25)
vector_store = QdrantVectorStore(
    client=client, 
    collection_name="documentos_rag",
    enable_hybrid=True
)

# ==========================================
# 4. MEMORIA MULTI-USUARIO
# ==========================================
motores_de_chat = {}


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
        return {
            "modelos": modelos,
            "modelo_actual": DEFAULT_LLM_MODEL,
        }
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
    texto_normalizado = texto.strip()
    return not texto_normalizado or texto_normalizado == "Empty Response"


def construir_system_prompt(prompt_personalizado: str | None = None) -> str:
    if prompt_personalizado:
        return prompt_personalizado

    return (
        "Eres un asistente experto en analisis documental. Responde siempre en espanol usando "
        "UNICAMENTE la informacion recuperada de los documentos indexados. "
        "Si la evidencia recuperada no es suficiente o no responde la pregunta, indica claramente "
        "que no tienes informacion suficiente en los documentos. "
        "No inventes datos, nombres, fechas, cifras ni conclusiones. "
        "Cuando respondas con informacion factual, cita de forma explicita el nombre del archivo "
        "del que sale el dato. Si varias fuentes apoyan la respuesta, integralas de forma breve y clara."
    )

# ==========================================
# 5. HORA ACTUAL (Conocimiento Temporal)
# ==========================================
def get_actual_time() -> str:
    """Obtiene la hora actual con formato de legibilidad."""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def detecte_consulta_hora(pregunta: str) -> bool:
    """Detecta si la consulta del usuario es sobre la hora."""
    palabras_hora = ["hora", "tiempo", "fecha", "fecha y hora", 
                      "qué hora", "qué tiempo", "qué fecha", 
                      "actual", "ahora", "mañana", "tarde", "tardeo", "noche"]
    pregunta_lower = pregunta.lower()
    return any(palabra in pregunta_lower for palabra in palabras_hora)

def inicializar_y_cargar_datos(
    directorio_docs: str = "./data/docs",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> str:
    if not os.path.exists(directorio_docs) or not os.listdir(directorio_docs):
        return "El directorio está vacío o no existe."

    configuracion_rag = construir_configuracion_rag(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documentos = SimpleDirectoryReader(directorio_docs).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    parser = crear_text_splitter(configuracion_rag)
    nodos = parser.get_nodes_from_documents(documentos)

    # Al crear los nodos, Qdrant genera vectores + palabras clave automáticamente.
    VectorStoreIndex(nodes=nodos, storage_context=storage_context)
    
    global motores_de_chat
    motores_de_chat.clear()
    
    return (
        f"Se han indexado {len(documentos)} documentos en {len(nodos)} fragmentos "
        f"(chunk_size={configuracion_rag['chunk_size']}, overlap={configuracion_rag['chunk_overlap']})."
    )

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
    global motores_de_chat
    contenido_generado = []
    
    try:
        # 1. Respuesta directa sobre la hora (sin consultar documentos)
        if detecte_consulta_hora(pregunta):
            hora_actual = get_actual_time()
            fecha_actual = datetime.now().strftime("%d/%m/%Y")
            
            yield f"🕐 Hora actual: {hora_actual}"
            yield f"📅 Fecha actual: {fecha_actual}"
            yield "\n\n"
            if not detecte_consulta_hora(pregunta):
                yield "¿Te refieres a algo más? Si es sobre los documentos indexados, voy a buscar esa información por ti."
            else:
                yield " Puedo mostrarte la hora o darte información sobre los documentos que tengo indexados."
            
            return
        
        system_prompt = construir_system_prompt(system_prompt)
        modelo_a_usar = llm_model or DEFAULT_LLM_MODEL
        configuracion_rag = construir_configuracion_rag(
            similarity_top_k=similarity_top_k,
            similarity_cutoff=similarity_cutoff,
            source_top_k=source_top_k,
            usar_reranker=usar_reranker,
            rerank_top_n=rerank_top_n,
        )
        clave_configuracion = crear_clave_configuracion(configuracion_rag)

        recreate = False
        motor_data = motores_de_chat.get(session_id)
        if not motor_data:
            recreate = True
        else:
            if (
                motor_data.get("system_prompt") != system_prompt
                or motor_data.get("temperature") != temperature
                or motor_data.get("rag_config") != clave_configuracion
                or motor_data.get("llm_model") != modelo_a_usar
            ):
                recreate = True

        if recreate:
            custom_llm = OpenAI(
                api_base=openai_api_base,
                api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
                model=modelo_a_usar,
                temperature=temperature
            )
            
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            postprocesadores = crear_postprocesadores(configuracion_rag)
            
            nuevo_motor = index.as_chat_engine(
                chat_mode="context",
                llm=custom_llm,
                system_prompt=system_prompt,
                similarity_top_k=configuracion_rag["similarity_top_k"],
                node_postprocessors=postprocesadores
            )
            
            motores_de_chat[session_id] = {
                "engine": nuevo_motor,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "rag_config": clave_configuracion,
                "llm_model": modelo_a_usar,
            }
        
        motor_usuario = motores_de_chat[session_id]["engine"]
        respuesta_streaming = motor_usuario.stream_chat(pregunta)
        
        buffer = ""
        in_think = False
        
        for texto_parcial in respuesta_streaming.response_gen:
            buffer += texto_parcial
            
            if not in_think:
                if "<think>" in buffer:
                    partes = buffer.split("<think>", 1)
                    if partes[0]:
                        contenido_generado.append(partes[0])
                    yield partes[0]
                    yield "\n\n> 💭 **Razonamiento de la IA:**\n> "
                    buffer = partes[1]
                    in_think = True
                else:
                    idx = buffer.rfind("<")
                    if idx != -1 and len(buffer) - idx < 7:
                        yield buffer[:idx]
                        if buffer[:idx]:
                            contenido_generado.append(buffer[:idx])
                        buffer = buffer[idx:]
                    else:
                        yield buffer
                        if buffer:
                            contenido_generado.append(buffer)
                        buffer = ""
            
            if in_think:
                if "</think>" in buffer:
                    partes = buffer.split("</think>", 1)
                    yield partes[0].replace("\n", "\n> ")
                    yield "\n\n---\n\n"
                    buffer = partes[1]
                    in_think = False
                else:
                    idx = buffer.rfind("</")
                    if idx != -1 and len(buffer) - idx < 8:
                        yield buffer[:idx].replace("\n", "\n> ")
                        buffer = buffer[idx:]
                    else:
                        yield buffer.replace("\n", "\n> ")
                        buffer = ""

        if buffer:
            if in_think:
                yield buffer.replace("\n", "\n> ")
            else:
                contenido_generado.append(buffer)
                yield buffer

        respuesta_texto = "".join(contenido_generado).strip()
        if es_respuesta_vacia(respuesta_texto):
            yield (
                "No encontré evidencia suficiente en los fragmentos recuperados para responder con fiabilidad. "
                "Prueba a bajar el filtro de relevancia o a reindexar con fragmentos más pequeños."
            )
            return

        # Añadido de citas de fuentes
        if hasattr(respuesta_streaming, "source_nodes") and respuesta_streaming.source_nodes:
            fuentes_str = "\n\n---\n\n**📚 Fuentes consultadas:**\n"
            docs_citados = set()
            for nodo in respuesta_streaming.source_nodes[:configuracion_rag["source_top_k"]]:
                archivo = nodo.node.metadata.get("file_name", "Desconocido")
                pagina = nodo.node.metadata.get("page_label", "N/A")
                score = getattr(nodo, "score", 0.0)
                
                doc_id = f"- 📄 `{archivo}` (Pág: {pagina}) - Relevancia: {score:.2f}"
                if doc_id not in docs_citados:
                    docs_citados.add(doc_id)
                    fuentes_str += doc_id + "\n"
                    
            if docs_citados:
                yield fuentes_str
            
    except Exception as e:
        yield f"Error al consultar el modelo: {str(e)}"
