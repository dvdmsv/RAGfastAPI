import os
import streamlit as st
import requests
import uuid

# URL de nuestro backend FastAPI
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG IA Local", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot RAG Multi-Hilo")

# ---------------------------------------------------------
# 1. GESTIÓN DE HILOS MÚLTIPLES (Memoria del Frontend)
# ---------------------------------------------------------
if "chats" not in st.session_state:
    primer_id = str(uuid.uuid4())
    st.session_state.chats = {
        primer_id: {"titulo": "Chat 1", "mensajes": []}
    }
    st.session_state.chat_actual_id = primer_id

chat_actual_id = st.session_state.chat_actual_id

# Si por alguna razón el chat actual fue borrado, forzamos un reset de seguridad
if chat_actual_id not in st.session_state.chats:
    st.session_state.chat_actual_id = list(st.session_state.chats.keys())[0]
    chat_actual_id = st.session_state.chat_actual_id

chat_actual = st.session_state.chats[chat_actual_id]

# ---------------------------------------------------------
# 2. BARRA LATERAL (Gestión de Hilos y Documentos)
# ---------------------------------------------------------
with st.sidebar:
    st.header("💬 Mis Conversaciones")
    
    # Botón para crear un nuevo hilo
    if st.button("➕ Iniciar Nuevo Chat", use_container_width=True, type="primary"):
        nuevo_id = str(uuid.uuid4())
        # Buscamos un número de chat que no exista para el título
        numero_chat = len(st.session_state.chats) + 1
        st.session_state.chats[nuevo_id] = {"titulo": f"Chat {numero_chat}", "mensajes": []}
        st.session_state.chat_actual_id = nuevo_id
        st.rerun()

    st.write("---")
    
    # Listamos los botones de cada chat con la opción de eliminarlos
    # Usamos list() para evitar errores al modificar el diccionario mientras lo iteramos
    for cid, datos_chat in list(st.session_state.chats.items()):
        # Dividimos el espacio: 80% para el nombre del chat, 20% para la papelera
        col1, col2 = st.columns([8, 2])
        
        with col1:
            if cid == chat_actual_id:
                st.button(f"👉 {datos_chat['titulo']}", key=f"sel_{cid}", use_container_width=True, disabled=True)
            else:
                if st.button(f"📄 {datos_chat['titulo']}", key=f"sel_{cid}", use_container_width=True):
                    st.session_state.chat_actual_id = cid
                    st.rerun()
                    
        with col2:
            # Botón de eliminar
            if st.button("🗑️", key=f"del_{cid}"):
                # Borramos el chat de la memoria
                del st.session_state.chats[cid]
                
                # Lógica de seguridad: si borramos el último chat, creamos uno nuevo vacío
                if len(st.session_state.chats) == 0:
                    nuevo_id = str(uuid.uuid4())
                    st.session_state.chats[nuevo_id] = {"titulo": "Chat 1", "mensajes": []}
                    st.session_state.chat_actual_id = nuevo_id
                # Si borramos el chat en el que estábamos, saltamos al primero disponible
                elif cid == chat_actual_id:
                    st.session_state.chat_actual_id = list(st.session_state.chats.keys())[0]
                
                st.rerun()
                
    st.write("---")

    st.header("📂 Base de Conocimiento")
    archivo_subido = st.file_uploader("Sube un archivo (TXT, PDF)", type=["txt", "pdf"])
    
    if st.button("Subir e Indexar"):
        if archivo_subido is not None:
            with st.spinner('Procesando y vectorizando...'):
                archivos = {"file": (archivo_subido.name, archivo_subido.getvalue())}
                respuesta = requests.post(f"{API_URL}/api/upload", files=archivos)
                
                if respuesta.status_code == 200:
                    st.success("¡Archivo aprendido correctamente!")
                else:
                    st.error("Error al subir el archivo.")
        else:
            st.warning("Por favor, selecciona un archivo primero.")

    st.write("---")
    
    st.header("📚 Archivos en Memoria")
    
    try:
        respuesta_docs = requests.get(f"{API_URL}/api/documents")
        
        if respuesta_docs.status_code == 200:
            documentos = respuesta_docs.json().get("documentos", [])
            
            if documentos:
                with st.expander(f"Ver documentos indexados ({len(documentos)})", expanded=True):
                    # Iteramos sobre cada archivo
                    for doc in documentos:
                        # Dividimos el espacio: 85% texto, 15% botón
                        col_doc, col_del = st.columns([8.5, 1.5])
                        
                        with col_doc:
                            st.text(f"📄 {doc}")
                            
                        with col_del:
                            # Creamos un botón de borrado único para cada archivo
                            if st.button("🗑️", key=f"del_file_{doc}", help="Eliminar documento"):
                                # Llamamos a nuestro nuevo endpoint DELETE
                                resp_borrar = requests.delete(f"{API_URL}/api/documents/{doc}")
                                
                                if resp_borrar.status_code == 200:
                                    st.toast(f"Archivo {doc} eliminado.")
                                    st.rerun() # Recargamos la interfaz al instante
                                else:
                                    st.error("Error al borrar.")
            else:
                st.info("La base de conocimiento está vacía. ¡Sube tu primer archivo!")
        else:
            st.error("No se pudo cargar la lista.")
    except Exception:
        st.warning("FastAPI no está respondiendo. ¿Está encendido el servidor?")

# ---------------------------------------------------------
# 3. INTERFAZ DE CHAT PRINCIPAL
# ---------------------------------------------------------
for mensaje in chat_actual["mensajes"]:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    with st.chat_message("user"):
        st.markdown(prompt)
        
    chat_actual["mensajes"].append({"rol": "user", "contenido": prompt})
    
    with st.chat_message("assistant"):
        try:
            payload = {
                "pregunta": prompt,
                "session_id": chat_actual_id
            }
            
            respuesta_api = requests.post(f"{API_URL}/api/chat", json=payload, stream=True)
            
            if respuesta_api.status_code == 200:
                def leer_flujo():
                    for chunk in respuesta_api.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk
                
                respuesta_completa = st.write_stream(leer_flujo())
                chat_actual["mensajes"].append({"rol": "assistant", "contenido": respuesta_completa})
            else:
                st.error("Error de conexión con el Backend.")
        except Exception as e:
            st.error(f"Asegúrate de que FastAPI esté corriendo. Error: {e}")