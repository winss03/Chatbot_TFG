from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import mysql.connector
import os
import logging
from dotenv import load_dotenv
import threading
import time

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la base de datos usando variables de entorno
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', '3306'))
}

# Diccionario de alias para los dispositivos
ALIAS_DISPOSITIVOS = {
    "354018114174919": "5091HSS",
    "350612079181804": "2233NBC",
    "350317175357569": "1452MXC",
    "350612076368818": "0824LRR"
}


def cargar_datos():
    """Carga y procesa datos de la tabla mensajes"""
    try:
        logger.info("📚 Conectando a la base de datos MySQL...")
        logger.debug(f"DB_CONFIG: {DB_CONFIG}")
        conn = mysql.connector.connect(**DB_CONFIG)
        logger.info("✅ Conexión exitosa a MySQL")
        cursor = conn.cursor(dictionary=True)

        # Consulta principal
        query = """
        SELECT 
            id,
            device_id,
            sat,
            mov_state,
            voltaje/1000.0 as voltaje_v,
            volt_bateria/1000.0 as bateria_v,
            temperatura,
            fecha_recepcion,
            lat, lng, alt
        FROM mensajes
        ORDER BY fecha_recepcion DESC
        """

        cursor.execute(query)
        registros = cursor.fetchall()

        # Convertir a objetos Document directamente
        documents = []

        # Procesar por dispositivo
        dispositivos = {}
        for reg in registros:
            if reg['device_id'] not in dispositivos:
                dispositivos[reg['device_id']] = []
            dispositivos[reg['device_id']].append(reg)

        # Crear documentos por dispositivo
        for device_id, regs in dispositivos.items():
            alias = ALIAS_DISPOSITIVOS.get(device_id, device_id)
            ultimo_reg = regs[0]  # El más reciente

            # Comprobar valores nulos
            lat = ultimo_reg['lat'] if ultimo_reg['lat'] is not None else "No disponible"
            lng = ultimo_reg['lng'] if ultimo_reg['lng'] is not None else "No disponible"
            alt = ultimo_reg['alt'] if ultimo_reg['alt'] is not None else "No disponible"
            mov_state = 'En movimiento' if ultimo_reg['mov_state'] == 1 else (
                'Detenido' if ultimo_reg['mov_state'] == 0 else "No disponible")
            sat = ultimo_reg['sat'] if ultimo_reg['sat'] is not None else "No disponible"
            voltaje_v = f"{ultimo_reg['voltaje_v']:.2f}V" if ultimo_reg['voltaje_v'] is not None else "No disponible"
            bateria_v = f"{ultimo_reg['bateria_v']:.2f}V" if ultimo_reg['bateria_v'] is not None else "No disponible"
            temperatura = f"{ultimo_reg['temperatura']}°C" if ultimo_reg['temperatura'] is not None else "No disponible"
            fecha_recepcion = ultimo_reg['fecha_recepcion'] if ultimo_reg['fecha_recepcion'] is not None else "No disponible"

            resumen = f"""📱 Dispositivo {alias} ({device_id}):
                Estado actual:
                • Ubicación: {lat}, {lng}
                • Altitud: {alt} metros
                • Movimiento: {mov_state}
                • Satélites: {sat}
                • Voltaje: {voltaje_v}
                • Batería: {bateria_v}
                • Temperatura: {temperatura}
                • Última actualización: {fecha_recepcion}"""

            # Crear objeto Document directamente
            from langchain.schema import Document
            doc = Document(
                page_content=resumen,
                metadata={
                    "tipo": "estado_actual",
                    "device_id": device_id,
                    "alias": alias,
                    "timestamp": str(ultimo_reg['fecha_recepcion'])
                }
            )
            documents.append(doc)

        logger.info(
            f"✅ Procesados {len(documents)} documentos de {len(dispositivos)} dispositivos")
        return documents

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.debug("Error detallado:", exc_info=True)
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()


def configurar_chatbot(documents):
    """Configura el chatbot con los documentos procesados"""
    try:
        logger.info(f"🔄 Configurando chatbot con {len(documents)} documentos")

        # 1. Crear embeddings
        embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="embed-multilingual-v3.0"
        )

        # 2. Crear base de datos vectorial directamente con los documentos
        vectorstore = FAISS.from_documents(documents, embeddings)

        # 3. Configurar memoria
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=15  # Mantener las últimas 15 interacciones
        )

        # 4. Crear el modelo con configuración en español
        llm = ChatCohere(
            temperature=0.7,
            model="command-r",
            max_tokens=20,
            prompt_prefix="""Eres un asistente especializado en interpretar información de bases de datos. 
                Eres un asistente amable y conciso que responde siempre en español con fraseshola claras."""
        )

        # 5. Crear el prompt template
        prompt = PromptTemplate(
            template="""Eres un asistente especializado en analizar bases de datos de dispositivos IoT.
            Cada fila de la base representa un mensaje enviado por un dispositivo, incluyendo datos como ubicación, voltaje, batería, temperatura, fecha, estado de movimiento y más.

            🔹 Tu tarea es responder SIEMPRE en español de forma:
            - Breve (hasta 4 o 5 líneas)
            - Clara, útil y amable
            - Natural y cercana, como si hablaras con una persona
            - Puedes usar saludos cortos o frases de cortesía si es apropiado
            - Directa y precisa, enfocándote en lo que se pregunta
            - Sin repetir la pregunta ni incluir explicaciones innecesarias

            🔸 Instrucciones específicas:
            - Si preguntan "dame información del dispositivo X" → Da el estado actual incluyendo:
                • Satélites conectados (sat)
                • Estado de movimiento (mov_state: 0 = Detenido, 1 = En movimiento)
                • Voltaje del sistema (voltaje_v)
                • Voltaje de batería (bateria_v)
                • Temperatura en °C
                • Fecha y hora del último mensaje (fecha_recepcion)
                • Ubicación: latitud, longitud, altitud, y ciudad/pueblo cercano si se puede

            - Si preguntan "¿Qué o Cuántos dispositivos hay?" → Responde con los alias y device_id separados por comas, puedes añadir una frase introductoria breve.

            - Si preguntan "Ubicación del dispositivo X" → Da el device_id con latitud, longitud y una referencia aproximada a la ciudad o pueblo más cercano y la Fecha y hora del último mensaje .

            - Si preguntan "¿Está en movimiento?" → Responde “Sí” o “No”, incluyendo el ID si es relevante, y puedes añadir una frase breve adicional.

            - Si preguntan "¿Cuándo fue el último mensaje?" → Da solo la fecha y hora exacta, puedes añadir una frase breve si lo ves adecuado.

            - Si preguntan "Voltaje o batería del dispositivo X" → Responde con el device_id y el valor en voltios con unidad.

            - - Si preguntan "¿Qué temperatura tiene el dispositivo?" → Responde con el alias y device_id, el valor de temperatura y la unidad °C. Si la temperatura no está disponible para el último registro, busca el registro más reciente que sí la tenga y responde con ese valor, indicando también la fecha y hora de ese registro. Si no hay ningún dato de temperatura disponible, indícalo amablemente.

            ❗ Si la consulta no tiene sentido, es irreconocible, está en otro idioma, o no está relacionada con dispositivos, responde amablemente: "Lo siento, no entiendo tu pregunta. ¿Puedes reformularla o preguntar algo sobre los dispositivos?".
            ❗ No expliques tu estilo de respuesta ni menciones que estás siguiendo instrucciones.

            Contexto actual: {context}
            Pregunta: {question}

            Responde:
            """,
            input_variables=["context", "question"]
        )

        # 6. Crear la cadena de conversación
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False  # Desactivamos verbose para evitar logs excesivos
        )

        logger.info("✅ Chatbot configurado correctamente")
        return qa_chain

    except Exception as e:
        logger.error(f"❌ Error al configurar el chatbot: {e}")
        logger.debug("Error detallado:", exc_info=True)
        return None


def chatbot():
    """Función principal del chatbot interactivo"""
    # Cargar datos de la base de datos
    chunks = cargar_datos()  # Cambiamos cargar_documento() por cargar_datos()
    if not chunks:
        logger.error("❌ No se pudieron cargar los datos.")
        return

    # Configurar el chatbot
    qa_chain = configurar_chatbot(chunks)
    if not qa_chain:
        logger.error("❌ No se pudo configurar el chatbot.")
        return

    # Iniciar conversación
    logger.info(
        "\n 🤖 Chatbot listo. Escribe tu consulta o 'salir' para finalizar la conversación.")

    while True:
        pregunta = input("\n👤 Tu consulta: ").strip()

        if not pregunta:
            logger.info("❌ Por favor, ingresa una pregunta válida.")
            continue

        if pregunta.lower() == 'salir':
            logger.info("👋 ¡Gracias por usar el chatbot! Hasta pronto.")
            break

        try:

            result = qa_chain.invoke({"question": pregunta})
            respuesta = result['answer']

            # Mostrar respuesta
            print("\n🤖 Respuesta:")
            print(respuesta)

        except Exception as e:
            logger.error(f"❌ Error al procesar la pregunta: {str(e)}")
            logger.debug("Error detallado:", exc_info=True)

# nombre del script para evitar recargas innecesarias
if __name__ == "__main__":
    chatbot()

### Código para la API REST ###
chatbot_chain = None  # Variable global para mantener la instancia

## Función para inicializar el chatbot al arrancar la API
def inicializar_chatbot():
    global chatbot_chain
    chunks = cargar_datos()
    if not chunks:
        raise Exception("No se pudieron cargar los datos.")

    chatbot_chain = configurar_chatbot(chunks)
    if not chatbot_chain:
        raise Exception("No se pudo configurar el chatbot.")

    return True

## Función para responder preguntas a través de la API
def responder_pregunta(pregunta):
    if not chatbot_chain:
        raise Exception("Chatbot no inicializado.")

    result = chatbot_chain.invoke({"question": pregunta})
    return result['answer']

# Función para recargar automáticamente el chatbot cada cierto tiempo
# def recarga_periodica_chatbot(intervalo_segundos=3600):
#     """Recarga el chatbot cada 'intervalo_segundos' segundos (por defecto, cada hora)."""
#     while True:
#         try:
#             logger.info("🔄 Recargando datos del chatbot automáticamente...")
#             inicializar_chatbot()
#             logger.info("✅ Chatbot recargado correctamente.")
#         except Exception as e:
#             logger.error(f"❌ Error al recargar el chatbot: {e}")
#         time.sleep(intervalo_segundos)

# Iniciar el hilo de recarga automática solo si es el proceso principal
# if __name__ != "__main__":
#     hilo_recarga = threading.Thread(target=recarga_periodica_chatbot, args=(3600,), daemon=True)
#     hilo_recarga.start()
    
#         raise Exception("Chatbot no inicializado.")
    
#     result = chatbot_chain.invoke({"question": pregunta})
#     return result['answer']
