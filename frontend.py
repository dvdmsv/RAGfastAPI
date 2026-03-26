import os
import uuid
import json
from pathlib import Path

import requests
import streamlit as st

# URL de nuestro backend FastAPI
API_URL = os.getenv("API_URL", "http://localhost:8000")
PROMPT_DEFECTO = (
    "Eres un asistente experto en analisis documental. Responde siempre en espanol usando "
    "UNICAMENTE la informacion recuperada de los documentos indexados. "
    "Si la evidencia recuperada no es suficiente o no responde la pregunta, indica claramente "
    "que no tienes informacion suficiente en los documentos. "
    "No inventes datos, nombres, fechas, cifras ni conclusiones. "
    "Cuando respondas con informacion factual, cita de forma explicita el nombre del archivo del que sale el dato."
)
THINKING_HEADER = "> 💭 **Razonamiento de la IA:**"
THINKING_HINTS = (
    THINKING_HEADER,
    "> 💭",
    "Razonamiento de la IA",
)
STATE_FILE = Path("data/frontend_state.json")
PRESETS_RAG = {
    "Chat": {
        "temperature": 0.2,
        "chunk_size": 700,
        "chunk_overlap": 120,
        "similarity_top_k": 6,
        "similarity_cutoff": 0.30,
        "source_top_k": 4,
        "usar_reranker": False,
        "rerank_top_n": 3,
    },
    "Precision": {
        "temperature": 0.1,
        "chunk_size": 550,
        "chunk_overlap": 120,
        "similarity_top_k": 5,
        "similarity_cutoff": 0.40,
        "source_top_k": 3,
        "usar_reranker": True,
        "rerank_top_n": 2,
    },
    "Rapido": {
        "temperature": 0.1,
        "chunk_size": 900,
        "chunk_overlap": 100,
        "similarity_top_k": 4,
        "similarity_cutoff": 0.25,
        "source_top_k": 3,
        "usar_reranker": False,
        "rerank_top_n": 2,
    },
    "Exploracion": {
        "temperature": 0.2,
        "chunk_size": 800,
        "chunk_overlap": 150,
        "similarity_top_k": 8,
        "similarity_cutoff": 0.20,
        "source_top_k": 5,
        "usar_reranker": True,
        "rerank_top_n": 4,
    },
}


def cargar_estado_persistido() -> dict:
    if not STATE_FILE.exists():
        return {}

    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def guardar_estado_persistido() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    estado = {
        "chats": st.session_state.get("chats", {}),
        "chat_actual_id": st.session_state.get("chat_actual_id"),
        "rag_settings": st.session_state.get("rag_settings", {}),
        "mostrar_thinking": st.session_state.get("mostrar_thinking", True),
        "auto_expandir_thinking": st.session_state.get("auto_expandir_thinking", False),
    }
    STATE_FILE.write_text(json.dumps(estado, ensure_ascii=True, indent=2), encoding="utf-8")


def aplicar_preset_rag(nombre_preset: str) -> None:
    preset = PRESETS_RAG[nombre_preset]
    st.session_state.rag_settings.update(preset)
    guardar_estado_persistido()


def obtener_modelos_disponibles() -> tuple[list[str], str | None, str | None]:
    try:
        respuesta = requests.get(f"{API_URL}/api/models", timeout=10)
        respuesta.raise_for_status()
        payload = respuesta.json()
        return payload.get("modelos", []), payload.get("modelo_actual"), payload.get("error")
    except requests.RequestException as exc:
        return [], None, f"No se pudo consultar LM Studio: {exc}"


def filtrar_modelos_chat(modelos: list[str]) -> list[str]:
    palabras_no_chat = (
        "embedding",
        "embed",
        "text-embedding",
        "nomic-embed",
        "bge",
        "e5",
    )
    return [
        modelo for modelo in modelos
        if not any(palabra in modelo.lower() for palabra in palabras_no_chat)
    ]


def separar_respuesta_y_thinking(contenido: str) -> tuple[str, str | None, str | None]:
    if THINKING_HEADER not in contenido:
        return contenido, None, None

    partes = contenido.split(THINKING_HEADER, 1)
    intro = partes[0].strip()
    resto = partes[1]

    if "\n\n---\n\n" in resto:
        think_raw, cierre = resto.split("\n\n---\n\n", 1)
    else:
        think_raw, cierre = resto, ""

    thinking = think_raw.replace("\n> ", "\n").strip()
    final = cierre.strip()
    return intro, thinking or None, final or None


def extraer_fuentes(cierre: str | None) -> tuple[str | None, list[str]]:
    if not cierre or "**📚 Fuentes consultadas:**" not in cierre:
        return cierre, []

    cuerpo, bloque_fuentes = cierre.split("**📚 Fuentes consultadas:**", 1)
    fuentes = []
    for linea in bloque_fuentes.splitlines():
        linea = linea.strip()
        if linea.startswith("- "):
            fuentes.append(linea[2:])

    cuerpo_limpio = cuerpo.strip() or None
    return cuerpo_limpio, fuentes


def construir_contenido_visible(contenido: str, mostrar_thinking: bool) -> tuple[str, bool]:
    respuesta, thinking, cierre = separar_respuesta_y_thinking(contenido)
    if not thinking:
        return contenido, False

    if mostrar_thinking:
        return contenido, True

    cierre_texto, fuentes = extraer_fuentes(cierre)
    bloques = []
    if respuesta:
        bloques.append(respuesta)
    if cierre_texto:
        bloques.append(f"---\n\n{cierre_texto}")
    if fuentes:
        fuentes_markdown = "\n".join(f"- {fuente}" for fuente in fuentes)
        bloques.append(f"---\n\n**📚 Fuentes consultadas:**\n{fuentes_markdown}")

    return "\n\n".join(bloques).strip(), True


def detectar_thinking_en_stream(contenido: str) -> bool:
    return any(pista in contenido for pista in THINKING_HINTS)


def renderizar_fuentes(fuentes: list[str]) -> None:
    if not fuentes:
        return

    with st.expander(f"Fuentes consultadas ({len(fuentes)})", expanded=False):
        st.info("Estas referencias son los documentos recuperados para construir la respuesta.")
        for fuente in fuentes:
            st.markdown(f"- {fuente}")


def renderizar_mensaje_chat(rol: str, contenido: str) -> None:
    with st.chat_message(rol):
        if rol != "assistant":
            st.markdown(contenido)
            return

        respuesta, thinking, cierre = separar_respuesta_y_thinking(contenido)
        cierre_texto, fuentes = extraer_fuentes(cierre)
        if respuesta:
            st.markdown(respuesta)
        if thinking:
            if st.session_state.get("mostrar_thinking", True):
                st.info(
                    "Modo think detectado. El modelo ha generado razonamiento interno; puedes verlo o mantenerlo plegado."
                )
                with st.expander(
                    "Ver razonamiento del modelo",
                    expanded=st.session_state.get("auto_expandir_thinking", False),
                ):
                    st.markdown(thinking)
            else:
                st.caption("El modelo utilizo razonamiento interno. Esta oculto segun tu configuracion.")
        if cierre_texto:
            st.markdown(f"---\n\n{cierre_texto}")
        renderizar_fuentes(fuentes)

st.set_page_config(page_title="RAG IA Local", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot RAG Multi-Hilo")
estado_persistido = cargar_estado_persistido()

# ---------------------------------------------------------
# 1. GESTIÓN DE HILOS MÚLTIPLES (Memoria del Frontend)
# ---------------------------------------------------------
if "chats" not in st.session_state:
    chats_guardados = estado_persistido.get("chats")
    if chats_guardados:
        st.session_state.chats = chats_guardados
        st.session_state.chat_actual_id = estado_persistido.get("chat_actual_id") or list(chats_guardados.keys())[0]
    else:
        primer_id = str(uuid.uuid4())
        st.session_state.chats = {
            primer_id: {"titulo": "Chat 1", "mensajes": []}
        }
        st.session_state.chat_actual_id = primer_id

if "rag_settings" not in st.session_state:
    rag_settings_guardados = estado_persistido.get("rag_settings", {})
    st.session_state.rag_settings = {
        "system_prompt": PROMPT_DEFECTO,
        "temperature": 0.1,
        "chunk_size": 700,
        "chunk_overlap": 120,
        "similarity_top_k": 6,
        "similarity_cutoff": 0.30,
        "source_top_k": 5,
        "usar_reranker": False,
        "rerank_top_n": 3,
        "llm_model": None,
    }
    st.session_state.rag_settings.update(rag_settings_guardados)

if "mostrar_thinking" not in st.session_state:
    st.session_state.mostrar_thinking = estado_persistido.get("mostrar_thinking", True)

if "auto_expandir_thinking" not in st.session_state:
    st.session_state.auto_expandir_thinking = estado_persistido.get("auto_expandir_thinking", False)

if "modelos_lm" not in st.session_state:
    st.session_state.modelos_lm = []

if "modelo_actual_lm" not in st.session_state:
    st.session_state.modelo_actual_lm = None

if "error_modelos_lm" not in st.session_state:
    st.session_state.error_modelos_lm = None

chat_actual_id = st.session_state.chat_actual_id
rag_settings = st.session_state.rag_settings

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
        guardar_estado_persistido()
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
                
                guardar_estado_persistido()
                st.rerun()
                
    st.write("---")

    with st.expander("⚙️ Ajustes Avanzados", expanded=False):
        st.caption("Los ajustes de indexacion se aplican al subir o reindexar documentos.")
        st.markdown("**Conversacion**")
        st.caption("Presets rapidos")
        col_preset_1, col_preset_2 = st.columns(2)
        with col_preset_1:
            if st.button("Chat", use_container_width=True, key="preset_chat"):
                aplicar_preset_rag("Chat")
                st.rerun()
            if st.button("Rapido", use_container_width=True, key="preset_rapido"):
                aplicar_preset_rag("Rapido")
                st.rerun()
        with col_preset_2:
            if st.button("Precision", use_container_width=True, key="preset_precision"):
                aplicar_preset_rag("Precision")
                st.rerun()
            if st.button("Exploracion", use_container_width=True, key="preset_exploracion"):
                aplicar_preset_rag("Exploracion")
                st.rerun()
        st.caption("Chat: equilibrado. Precision: mas estricto. Rapido: menos latencia. Exploracion: mas contexto.")
        col_modelo, col_refrescar = st.columns([4, 1])
        with col_refrescar:
            if st.button("↻", key="refrescar_modelos", help="Recargar modelos disponibles", use_container_width=True):
                modelos, modelo_actual, error_modelos = obtener_modelos_disponibles()
                st.session_state.modelos_lm = filtrar_modelos_chat(modelos)
                st.session_state.modelo_actual_lm = modelo_actual
                st.session_state.error_modelos_lm = error_modelos
                guardar_estado_persistido()

        if not st.session_state.modelos_lm and st.session_state.error_modelos_lm is None:
            modelos, modelo_actual, error_modelos = obtener_modelos_disponibles()
            st.session_state.modelos_lm = filtrar_modelos_chat(modelos)
            st.session_state.modelo_actual_lm = modelo_actual
            st.session_state.error_modelos_lm = error_modelos

        with col_modelo:
            if st.session_state.modelos_lm:
                modelo_guardado = rag_settings.get("llm_model") or st.session_state.modelo_actual_lm or st.session_state.modelos_lm[0]
                indice_actual = st.session_state.modelos_lm.index(modelo_guardado) if modelo_guardado in st.session_state.modelos_lm else 0
                rag_settings["llm_model"] = st.selectbox(
                    "Modelo LLM",
                    options=st.session_state.modelos_lm,
                    index=indice_actual,
                    help="Selecciona el modelo a usar. LM Studio lo cargará automáticamente al enviar el siguiente mensaje.",
                )
            else:
                st.text_input(
                    "Modelo LLM",
                    value=st.session_state.modelo_actual_lm or "",
                    disabled=True,
                    help="No se han podido cargar los modelos disponibles desde LM Studio.",
                )
        st.caption("Pulsa ↻ para detectar los modelos disponibles en LM Studio y selecciona el que quieras usar.")
        if st.session_state.error_modelos_lm:
            st.warning(st.session_state.error_modelos_lm)
        st.session_state.mostrar_thinking = st.checkbox(
            "Mostrar bloques de razonamiento",
            value=st.session_state.mostrar_thinking,
            help="Muestra u oculta el bloque especial del razonamiento interno cuando el modelo responda con etiquetas think.",
        )
        st.caption("Activalo si quieres inspeccionar como razona el modelo. Desactivalo para una interfaz mas limpia.")
        st.session_state.auto_expandir_thinking = st.checkbox(
            "Abrir razonamiento automaticamente",
            value=st.session_state.auto_expandir_thinking,
            disabled=not st.session_state.mostrar_thinking,
            help="Si esta opcion esta activa, los bloques de razonamiento se desplegaran automaticamente al renderizar cada respuesta.",
        )
        st.caption("Recomendado desactivado para chat normal y activado cuando estes depurando respuestas.")
        rag_settings["system_prompt"] = st.text_area(
            "System Prompt",
            value=rag_settings["system_prompt"],
            height=150,
            help="Instrucciones base para el modelo. Define el tono, el idioma y las reglas de citacion que debe seguir al responder.",
        )
        rag_settings["temperature"] = st.slider(
            "Temperatura (creatividad)",
            min_value=0.0,
            max_value=1.0,
            value=float(rag_settings["temperature"]),
            step=0.1,
            help="Controla la creatividad del modelo. Valores bajos hacen respuestas mas estables y fieles; valores altos las vuelven mas variadas.",
        )
        st.caption("Recomendado para chat RAG: 0.0-0.2. Usa valores altos solo si quieres respuestas mas libres.")
        st.info("Estos parametros afectan al estilo y estabilidad de la respuesta.")

        st.markdown("**Recuperacion y Precision**")
        rag_settings["chunk_size"] = st.slider(
            "Tamano de fragmento",
            min_value=300,
            max_value=1500,
            value=int(rag_settings["chunk_size"]),
            step=50,
            help="Tamano de cada trozo en el que se divide un documento al indexarlo. Fragmentos pequenos mejoran precision; fragmentos grandes conservan mas contexto.",
        )
        st.caption("Recomendado: 500-900. Si las respuestas mezclan temas, baja este valor; si falta contexto, subelo.")
        rag_settings["chunk_overlap"] = st.slider(
            "Solape entre fragmentos",
            min_value=0,
            max_value=min(400, rag_settings["chunk_size"] // 2),
            value=min(int(rag_settings["chunk_overlap"]), max(0, rag_settings["chunk_size"] // 2)),
            step=10,
            help="Cantidad de texto compartido entre fragmentos consecutivos. Ayuda a no perder contexto cuando una idea cae entre dos trozos.",
        )
        st.caption("Recomendado: 80-150. Un solape bajo puede cortar ideas; uno muy alto añade redundancia.")
        rag_settings["similarity_top_k"] = st.slider(
            "Fragmentos recuperados",
            min_value=1,
            max_value=12,
            value=int(rag_settings["similarity_top_k"]),
            step=1,
            help="Numero de fragmentos candidatos que se recuperan antes de generar la respuesta. Subirlo aporta mas contexto, pero tambien mas ruido.",
        )
        st.caption("Recomendado: 4-8. Sube si el documento es largo o variado; baja si las respuestas traen demasiado ruido.")
        rag_settings["similarity_cutoff"] = st.slider(
            "Filtro de relevancia",
            min_value=0.0,
            max_value=1.0,
            value=float(rag_settings["similarity_cutoff"]),
            step=0.05,
            help="Umbral minimo de similitud para aceptar un fragmento. Un valor bajo deja pasar mas contexto; un valor alto filtra mas, pero puede dejar la respuesta vacia.",
        )
        st.caption("Recomendado para conversar: 0.25-0.35. Si aparece respuesta vacia, bajalo; si entra ruido, subelo un poco.")
        rag_settings["source_top_k"] = st.slider(
            "Fuentes mostradas",
            min_value=1,
            max_value=int(rag_settings["similarity_top_k"]),
            value=min(int(rag_settings["source_top_k"]), int(rag_settings["similarity_top_k"])),
            step=1,
            help="Cantidad maxima de fuentes que se muestran al final de la respuesta. Solo afecta a la presentacion, no a la recuperacion.",
        )
        st.caption("Recomendado: 3-5. Muestra suficientes referencias sin saturar la respuesta final.")
        st.success("Ajusta aqui cuando quieras mas precision, mas contexto o menos respuestas vacias.")

        st.markdown("**Rendimiento**")
        rag_settings["usar_reranker"] = st.checkbox(
            "Activar reranking semantico",
            value=bool(rag_settings["usar_reranker"]),
            help="Si el backend tiene soporte disponible, reordena los fragmentos recuperados para priorizar los mas utiles antes de generar la respuesta.",
        )
        st.caption("Activalo cuando el chat responda con contexto correcto pero mal priorizado. Puede mejorar precision a costa de algo mas de latencia.")
        rag_settings["rerank_top_n"] = st.slider(
            "Fragmentos tras reranking",
            min_value=1,
            max_value=int(rag_settings["similarity_top_k"]),
            value=min(int(rag_settings["rerank_top_n"]), int(rag_settings["similarity_top_k"])),
            step=1,
            disabled=not rag_settings["usar_reranker"],
            help="Numero de fragmentos finales que sobreviven tras el reranking. Menos fragmentos suelen dar respuestas mas enfocadas.",
        )
        st.caption("Recomendado: 2-4. Valores bajos enfocan la respuesta; valores altos conservan mas contexto.")
        st.warning("El reranking puede mejorar la calidad final, pero normalmente aumenta un poco la latencia.")
        guardar_estado_persistido()

    st.header("📂 Base de Conocimiento")
    archivo_subido = st.file_uploader("Sube un archivo (TXT, PDF)", type=["txt", "pdf"])
    
    if st.button("Subir e Indexar"):
        if archivo_subido is not None:
            with st.spinner('Procesando y vectorizando...'):
                archivos = {"file": (archivo_subido.name, archivo_subido.getvalue())}
                datos = {
                    "chunk_size": rag_settings["chunk_size"],
                    "chunk_overlap": rag_settings["chunk_overlap"],
                }
                respuesta = requests.post(f"{API_URL}/api/upload", files=archivos, data=datos)
                
                if respuesta.status_code == 200:
                    detalle = respuesta.json().get("estado_indexacion", "")
                    st.success(f"¡Archivo aprendido correctamente! {detalle}")
                else:
                    st.error(f"Error al subir el archivo: {respuesta.text}")
        else:
            st.warning("Por favor, selecciona un archivo primero.")

    if st.button("Reindexar con ajustes actuales", use_container_width=True):
        with st.spinner("Reconstruyendo indice con los ajustes seleccionados..."):
            payload_indexado = {
                "chunk_size": rag_settings["chunk_size"],
                "chunk_overlap": rag_settings["chunk_overlap"],
            }
            respuesta_indexado = requests.post(f"{API_URL}/api/indexar", json=payload_indexado)

            if respuesta_indexado.status_code == 200:
                st.success(respuesta_indexado.json().get("mensaje", "Indice reconstruido correctamente."))
            else:
                st.error(f"Error al reindexar: {respuesta_indexado.text}")

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
                                    guardar_estado_persistido()
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
    renderizar_mensaje_chat(mensaje["rol"], mensaje["contenido"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    with st.chat_message("user"):
        st.markdown(prompt)
        
    chat_actual["mensajes"].append({"rol": "user", "contenido": prompt})
    guardar_estado_persistido()
    
    with st.chat_message("assistant"):
        try:
            payload = {
                "pregunta": prompt,
                "session_id": chat_actual_id,
                "system_prompt": rag_settings["system_prompt"],
                "temperature": float(rag_settings["temperature"]),
                "similarity_top_k": int(rag_settings["similarity_top_k"]),
                "similarity_cutoff": float(rag_settings["similarity_cutoff"]),
                "source_top_k": int(rag_settings["source_top_k"]),
                "usar_reranker": bool(rag_settings["usar_reranker"]),
                "rerank_top_n": int(rag_settings["rerank_top_n"]),
                "llm_model": rag_settings.get("llm_model") or None,
            }
            
            respuesta_api = requests.post(f"{API_URL}/api/chat", json=payload, stream=True)
            
            if respuesta_api.status_code == 200:
                thinking_placeholder = st.empty()
                respuesta_placeholder = st.empty()
                respuesta_completa = ""
                think_detectado = False
                paso_animacion = 0

                for chunk in respuesta_api.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue

                    respuesta_completa += chunk
                    think_detectado = think_detectado or detectar_thinking_en_stream(respuesta_completa)
                    contenido_visible, think_detectado_actual = construir_contenido_visible(
                        respuesta_completa,
                        mostrar_thinking=st.session_state.mostrar_thinking,
                    )
                    think_detectado = think_detectado or think_detectado_actual

                    if contenido_visible:
                        respuesta_placeholder.markdown(contenido_visible)

                    if think_detectado and not st.session_state.mostrar_thinking:
                        puntos = "." * ((paso_animacion % 3) + 1)
                        thinking_placeholder.markdown(f"_Pensando{puntos}_")
                        paso_animacion += 1
                    else:
                        thinking_placeholder.empty()

                if think_detectado and not st.session_state.mostrar_thinking:
                    thinking_placeholder.markdown(
                        "_Razonamiento interno oculto. Activa su visualizacion en Ajustes Avanzados si quieres inspeccionarlo._"
                    )
                else:
                    thinking_placeholder.empty()

                chat_actual["mensajes"].append({"rol": "assistant", "contenido": respuesta_completa})
                guardar_estado_persistido()
                st.rerun()
            else:
                st.error("Error de conexión con el Backend.")
        except Exception as e:
            st.error(f"Asegúrate de que FastAPI esté corriendo. Error: {e}")
