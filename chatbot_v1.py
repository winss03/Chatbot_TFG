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

def cargar_datos():
    """Carga datos desde la base de datos MySQL"""
    try:
        # 1. Conectar a la base de datos
        logger.info("📚 Conectando a la base de datos MySQL...")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)  # Usar cursor de diccionario

        # 2. Obtener y mostrar las tablas disponibles
        cursor.execute("SHOW TABLES")
        tablas = cursor.fetchall()
        
        if not tablas:
            logger.error("❌ No se encontraron tablas en la base de datos")
            return None
            
        logger.info("\n📊 Tablas disponibles:")
        documents = []
        
        for tabla in tablas:
            nombre_tabla = list(tabla.values())[0]  # Obtener nombre de tabla
            logger.info(f"  - {nombre_tabla}")
            
            # Obtener datos de cada tabla
            cursor.execute(f"SELECT * FROM {nombre_tabla} LIMIT 10")
            rows = cursor.fetchall()
            
            # Crear documento para cada fila
            for row in rows:
                # Convertir cada fila a formato texto
                text = f"Tabla: {nombre_tabla}\n"
                for key, value in row.items():
                    text += f"{key}: {value}\n"
                documents.append({"page_content": text, "metadata": {"tabla": nombre_tabla}})

        # 3. Dividir el texto en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.create_documents([doc["page_content"] for doc in documents])
        logger.info(f"✅ Datos cargados y divididos en {len(chunks)} fragmentos")
        
        return chunks
        
    except mysql.connector.Error as e:
        logger.error(f"❌ Error MySQL: {e}")
        logger.error(f"  Código: {e.errno}")
        logger.error(f"  SQL State: {e.sqlstate}")
        logger.error(f"  Mensaje: {e.msg}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error general: {str(e)}")
        logger.debug("Error detallado:", exc_info=True)
        return None
        
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            logger.info("🔒 Conexión a la base de datos cerrada")

def configurar_chatbot(chunks):
    """Configura el chatbot con el documento procesado"""
    try:
        # 1. Crear embeddings
        embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="embed-multilingual-v3.0"
        )
        
        # 2. Crear base de datos vectorial
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # 3. Configurar memoria
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
        
        # 4. Crear el modelo con configuración en español
        llm = ChatCohere(
            temperature=0.4,
            model="command",
            max_tokens=500,
            prompt_prefix="""Eres un asistente especializado en analizar datos de dispositivos GPS,
            que siempre responde en español.Tus respuestas deben ser cortas, claras, precisas y en español."""
        )
        
        # 5. Crear el prompt template
        prompt = PromptTemplate(
            template="""Eres un asistente especializado en analizar datos de dispositivos GPS.
    
            Consideraciones sobre los datos:
            • Los timestamps están en formato Unix (milliseconds desde 1970)
            • Las coordenadas están en formato decimal (latitud, longitud)
            • Los voltajes se miden en milivoltios (convertir a voltios)
            • El estado de movimiento es binario (0=Detenido, 1=En movimiento)
            • Los ángulos están en grados (0-360)
            
            Contexto actual: {context}
            
            Pregunta: {question}
            
            Responde en español, incluyendo:
            - Unidades apropiadas para cada medida
            - Referencias temporales cuando sea relevante
            - Conversiones necesarias (ej: milivoltios a voltios)
            - Contexto geográfico cuando se mencionen ubicaciones""",
            input_variables=["context", "question"]
        )
        
        # 6. Crear la cadena de conversación
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
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
    logger.info("\n🤖 Chatbot CTFL iniciado. Escribe 'salir' para terminar.")
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
