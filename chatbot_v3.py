from langchain_cohere import ChatCohere
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import mysql.connector
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la base de datos
DB_CONFIG = {
    'host': 'test1.crs6as4q40wo.eu-west-2.rds.amazonaws.com',
    'user': 'admin',
    'password': '12345678',
    'database': 'mqtt_data'
}

# Configurar la API key de Cohere
os.environ["COHERE_API_KEY"] = "PDupltFxga8FYwjpE7t3UYZIBKRdpky4cpV8QpcF"

# Definición de tipos de datos y conversiones
COLUMNAS_MENSAJES = {
    'sat': {'tipo': 'int', 'descripcion': 'Número de satélites'},
    'mov_state': {'tipo': 'int', 'descripcion': 'Estado de movimiento (0=Detenido, 1=En movimiento)'},
    'voltaje': {'tipo': 'float', 'conversion': lambda x: x/1000.0, 'unidad': 'V'},
    'volt_bateria': {'tipo': 'float', 'conversion': lambda x: x/1000.0, 'unidad': 'V'},
    'temperatura': {'tipo': 'float', 'unidad': '°C'},
    'fecha_recepcion': {'tipo': 'datetime', 'formato': '%Y-%m-%d %H:%M:%S'}
}

def cargar_datos():
    """Carga y procesa datos de la tabla mensajes"""
    try:
        logger.info("📚 Conectando a la base de datos MySQL...")
        conn = mysql.connector.connect(**DB_CONFIG)
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
            ultimo_reg = regs[0]  # El más reciente
            resumen = f"""📱 Dispositivo {device_id}:
                Estado actual:
                • Ubicación: {ultimo_reg['lat']}, {ultimo_reg['lng']}
                • Altitud: {ultimo_reg['alt']} metros
                • Movimiento: {'En movimiento' if ultimo_reg['mov_state'] == 1 else 'Detenido'}
                • Satélites: {ultimo_reg['sat']}
                • Voltaje: {ultimo_reg['voltaje_v']:.2f}V
                • Batería: {ultimo_reg['bateria_v']:.2f}V
                • Temperatura: {ultimo_reg['temperatura']}°C
                • Última actualización: {ultimo_reg['fecha_recepcion']}"""

            # Crear objeto Document directamente
            from langchain.schema import Document
            doc = Document(
                page_content=resumen,
                metadata={
                    "tipo": "estado_actual",
                    "device_id": device_id,
                    "timestamp": str(ultimo_reg['fecha_recepcion'])
                }
            )
            documents.append(doc)

        logger.info(f"✅ Procesados {len(documents)} documentos de {len(dispositivos)} dispositivos")
        return documents

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
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
            k=15
        )
        
        # 4. Crear el modelo con configuración en español
        llm = ChatCohere(
            temperature=0.4,
            model="command",
            max_tokens=50,
            prompt_prefix="""Eres un asistente especializado en analizar datos de dispositivos GPS,
            que siempre responde en español.Tus respuestas deben ser cortas, claras, precisas y en español."""
        )
        
        # 5. Crear el prompt template
        prompt = PromptTemplate(
            template="""Eres un asistente especializado en analizar datos de dispositivos GPS.

            Información disponible por dispositivo:
            • Satélites conectados (sat)
            • Estado de movimiento (mov_state: 0=Detenido, 1=En movimiento)
            • Voltaje del sistema (en voltios)
            • Voltaje de batería (en voltios)
            • Temperatura (en °C)
            • Fecha y hora de recepción
            • Ubicación (latitud, longitud, altitud)

            Para responder preguntas:
            1. Si preguntan por ubicación:
            - Proporciona lat/lng del último registro
            - Incluye altitud si está disponible
            - Menciona la fecha de actualización

            2. Si preguntan por estado:
            - Indica si está en movimiento
            - Número de satélites conectados
            - Niveles de voltaje y batería
            - Temperatura actual

            3. Si preguntan por tendencias:
            - Compara con registros anteriores
            - Menciona cambios significativos
            - Indica períodos de tiempo

            Contexto actual: {context}
            Pregunta: {question}

            Responde en español, incluyendo siempre:
            - ID del dispositivo relevante
            - Timestamp del dato proporcionado
            - Unidades de medida apropiadas""",
            input_variables=["context", "question"]
        )
        
        # 6. Crear la cadena de conversación
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=True
        )
        
        logger.info("✅ Chatbot configurado correctamente en español")
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
    logger.info("\n🤖 Chatbot iniciado. Escribe 'salir' para terminar.")
    logger.info("💡 Puedes preguntarme cualquier cosa sobre el temario CTFL")
    
    while True:
        pregunta = input("\n👤 Tu pregunta: ").strip()
        
        if not pregunta:
            logger.info("❌ Por favor, ingresa una pregunta válida.")
            continue
            
        if pregunta.lower() == 'salir':
            logger.info("👋 ¡Hasta luego!")
            break
            
        try:
            # Usar invoke en lugar de __call__
            result = qa_chain.invoke({"question": pregunta})
            respuesta = result['answer']
            
            # Mostrar respuesta
            print("\n🤖 Respuesta:")
            print(respuesta)
            
        except Exception as e:
            logger.error(f"❌ Error al procesar la pregunta: {str(e)}")
            logger.debug("Error detallado:", exc_info=True)

if __name__ == "__main__":
    chatbot()
