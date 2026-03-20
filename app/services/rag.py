import os
import re
from datetime import datetime
from typing import Generator
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
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

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO DE LENGUAJE
# ==========================================
llm = OpenAI(
    api_base=openai_api_base,
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
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

# ==========================================
# 5. HORA ACTUAL (Conocimiento Temporal)
# ==========================================
def get_actual_time():
    """Obtiene la hora actual con formato de legibilidad."""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def detecte_consulta_hora(pregunta: str) -> bool:
    """Detecta si la consulta del usuario es sobre la hora."""
    palabras_hora = ["hora", "tiempo", "fecha", "fecha y hora", 
                      "qué hora", "qué tiempo", "qué fecha", 
                      "actual", "ahora", "mañana", "tarde", "tardeo", "noche"]
    pregunta_lower = pregunta.lower()
    return any(palabra in pregunta_lower for palabra in palabras_hora)

def inicializar_y_cargar_datos(directorio_docs: str = "./data/docs"):
    if not os.path.exists(directorio_docs) or not os.listdir(directorio_docs):
        return "El directorio está vacío o no existe."
        
    documentos = SimpleDirectoryReader(directorio_docs).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Al crear los documentos, Qdrant genera Vectores + Palabras clave automáticamente
    VectorStoreIndex.from_documents(documentos, storage_context=storage_context)
    
    global motores_de_chat
    motores_de_chat.clear()
    
    return f"Se han indexado {len(documentos)} fragmentos exitosamente."

def obtener_respuesta_rag_stream(
    pregunta: str, 
    session_id: str = "usuario_default",
    system_prompt: str = None,
    temperature: float = 0.1
) -> Generator[str, None, None]:
    global motores_de_chat
    
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
        
        if not system_prompt:
            system_prompt = (
                "Eres un asistente experto. Responde siempre en español basándote ÚNICAMENTE en los documentos proporcionados. "
                "REGLA DE ORO: Siempre que des un dato o información, DEBES citar explícitamente el nombre del archivo del cual proviene "
                "(por ejemplo: 'Según el documento reporte.pdf...' o 'Como se indica en el archivo resumen.txt'). "
                "Si la respuesta no está en los documentos, di simplemente que no tienes esa información."
            )

        recreate = False
        motor_data = motores_de_chat.get(session_id)
        if not motor_data:
            recreate = True
        else:
            if motor_data.get("system_prompt") != system_prompt or motor_data.get("temperature") != temperature:
                recreate = True
        
        if recreate:
            custom_llm = OpenAI(
                api_base=openai_api_base,
                api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                temperature=temperature
            )
            
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            
            nuevo_motor = index.as_chat_engine(
                chat_mode="context",
                llm=custom_llm,
                system_prompt=system_prompt,
                similarity_top_k=4
            )
            
            motores_de_chat[session_id] = {
                "engine": nuevo_motor,
                "system_prompt": system_prompt,
                "temperature": temperature
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
                    yield partes[0]
                    yield "\n\n> 💭 **Razonamiento de la IA:**\n> "
                    buffer = partes[1]
                    in_think = True
                else:
                    idx = buffer.rfind("<")
                    if idx != -1 and len(buffer) - idx < 7:
                        yield buffer[:idx]
                        buffer = buffer[idx:]
                    else:
                        yield buffer
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
                yield buffer

        # Añadido de citas de fuentes
        if hasattr(respuesta_streaming, "source_nodes") and respuesta_streaming.source_nodes:
            fuentes_str = "\n\n---\n\n**📚 Fuentes consultadas:**\n"
            docs_citados = set()
            for nodo in respuesta_streaming.source_nodes:
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
