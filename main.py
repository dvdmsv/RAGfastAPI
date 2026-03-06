import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional 
from fastapi.responses import StreamingResponse
from app.services.rag import inicializar_y_cargar_datos, obtener_respuesta_rag_stream

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

@app.get("/health")
async def health_check():
    return {"status": "ok", "mensaje": "Servidor operativo."}

# Eliminamos 'async' para que FastAPI use un hilo secundario y no bloquee el chat
@app.post("/api/upload")
def subir_documento(file: UploadFile = File(...)):
    """
    Recibe un documento (PDF, TXT), lo guarda en disco y avisa al sistema RAG.
    """
    directorio_destino = "./data/docs"
    os.makedirs(directorio_destino, exist_ok=True)
    
    ruta_archivo = os.path.join(directorio_destino, file.filename)
    
    try:
        with open(ruta_archivo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        resultado_indexacion = inicializar_y_cargar_datos(directorio_destino)
        
        return {
            "mensaje": f"Archivo '{file.filename}' subido con éxito.",
            "estado_indexacion": resultado_indexacion
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

# Eliminamos 'async' por la misma razón (evitar bloqueos al procesar vectores)
@app.post("/api/indexar")
def indexar_documentos():
    resultado = inicializar_y_cargar_datos()
    return {"mensaje": resultado}

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Devuelve la respuesta generada por la IA como un flujo continuo de texto (Streaming).
    """
    generador = obtener_respuesta_rag_stream(request.pregunta, request.session_id)
    return StreamingResponse(generador, media_type="text/event-stream")