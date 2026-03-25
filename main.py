import os
import shutil
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.rag import inicializar_y_cargar_datos, listar_modelos_lm_studio, obtener_respuesta_rag_stream

app = FastAPI(title="RAG API Local - IA & Big Data", version="1.0.0")

# Permitir a cualquier frontend comunicarse con nuestra API (Seguridad y CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción se ponen las URLs específicas del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    pregunta: str
    session_id: Optional[str] = "usuario_default"
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.1
    similarity_top_k: Optional[int] = None
    similarity_cutoff: Optional[float] = None
    source_top_k: Optional[int] = None
    usar_reranker: Optional[bool] = None
    rerank_top_n: Optional[int] = None


class IndexRequest(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

@app.get("/health")
async def health_check():
    return {"status": "ok", "mensaje": "Servidor operativo."}


@app.get("/api/models")
async def listar_modelos():
    return listar_modelos_lm_studio()

# Eliminamos 'async' para que FastAPI use un hilo secundario y no bloquee el chat
@app.post("/api/upload")
def subir_documento(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
):
    """
    Recibe un documento (PDF, TXT), lo guarda en disco y avisa al sistema RAG.
    """
    directorio_destino = "./data/docs"
    os.makedirs(directorio_destino, exist_ok=True)
    
    ruta_archivo = os.path.join(directorio_destino, file.filename)
    
    try:
        with open(ruta_archivo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        resultado_indexacion = inicializar_y_cargar_datos(
            directorio_destino,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        return {
            "mensaje": f"Archivo '{file.filename}' subido con éxito.",
            "estado_indexacion": resultado_indexacion
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

@app.get("/api/documents")
async def listar_documentos():
    """Devuelve la lista de archivos actualmente en la base de conocimiento."""
    directorio_docs = "./data/docs"
    
    # Si la carpeta no existe, devolvemos una lista vacía
    if not os.path.exists(directorio_docs):
        return {"documentos": []}
        
    try:
        # Leemos los nombres de los archivos que están en la carpeta
        archivos = [f for f in os.listdir(directorio_docs) if os.path.isfile(os.path.join(directorio_docs, f))]
        return {"documentos": archivos}
    except Exception as e:
        return {"error": str(e)}
    
@app.delete("/api/documents/{nombre_archivo}")
async def borrar_documento(nombre_archivo: str):
    """Borra un archivo físico y actualiza el índice RAG."""
    directorio_docs = "./data/docs"
    ruta_archivo = os.path.join(directorio_docs, nombre_archivo)
    
    # 1. Verificamos que el archivo exista por seguridad
    if not os.path.exists(ruta_archivo):
        return {"error": "El archivo no existe."}
        
    try:
        # 2. Borramos el archivo físico de la carpeta
        os.remove(ruta_archivo)
        
        # 3. Volvemos a indexar la carpeta para que Qdrant olvide el archivo borrado
        # Si la carpeta se queda vacía, la función lo manejará adecuadamente
        resultado_rag = inicializar_y_cargar_datos(directorio_docs)
        
        return {
            "mensaje": f"Archivo '{nombre_archivo}' eliminado con éxito.",
            "estado_bd": resultado_rag
        }
    except Exception as e:
        return {"error": f"Error al procesar el borrado: {str(e)}"}

# Eliminamos 'async' por la misma razón (evitar bloqueos al procesar vectores)
@app.post("/api/indexar")
def indexar_documentos(request: Optional[IndexRequest] = None):
    request = request or IndexRequest()
    resultado = inicializar_y_cargar_datos(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return {"mensaje": resultado}

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Devuelve la respuesta generada por la IA como un flujo continuo de texto (Streaming).
    """
    generador = obtener_respuesta_rag_stream(
        request.pregunta,
        request.session_id,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        similarity_top_k=request.similarity_top_k,
        similarity_cutoff=request.similarity_cutoff,
        source_top_k=request.source_top_k,
        usar_reranker=request.usar_reranker,
        rerank_top_n=request.rerank_top_n,
    )
    return StreamingResponse(generador, media_type="text/event-stream")
