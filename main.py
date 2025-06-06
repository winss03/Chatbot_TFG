import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from chatbot_v2 import inicializar_chatbot, responder_pregunta
from fastapi.responses import StreamingResponse
import time

# Configura el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot GPS API",
    description="API para consultar información de dispositivos GPS",
    version="2.4.0"
)

# Configuración de CORS actualizada para ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://8887-151-248-23-159.ngrok-free.app",
        "http://localhost:8000",
        "http://localhost"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class Pregunta(BaseModel):
    pregunta: str

    @validator('pregunta')
    def validar_pregunta(cls, v):
        if not v or not v.strip():
            raise ValueError('La pregunta no puede estar vacía')
        if len(v) > 500:
            raise ValueError('La pregunta es demasiado larga')
        return v

@app.on_event("startup")
def startup_event():
    logger.info("Inicializando chatbot...")
    inicializar_chatbot()
    logger.info("Chatbot inicializado correctamente")

@app.get("/")
def root():
    return {"mensaje": "¡Hola! Soy tu asistente de dispositivos GPS. ¿En qué puedo ayudarte hoy?"}

#opcion anterior para FastAPI sin streaming

@app.post("/preguntar")
def preguntar(p: Pregunta):
    logger.info(f"Pregunta recibida: {p.pregunta}")
    try:
        respuesta = responder_pregunta(p.pregunta)
        logger.info("Respuesta generada")
        return {"respuesta": respuesta}
    except Exception as e:
        logger.error(f"Error al responder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Permite conexiones externas
        port=8000,
        reload=True,
        ssl_keyfile=None,  # Ngrok maneja HTTPS
        ssl_certfile=None
    )
