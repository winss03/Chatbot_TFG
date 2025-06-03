# Usa una imagen base oficial de Python con soporte para ML
FROM python:3.11-slim

# Instala las dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requisitos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación
COPY . .

# Expone el puerto para la API
EXPOSE 8000

# Variables de entorno para el chatbot
ENV PYTHONUNBUFFERED=1
ENV COHERE_API_KEY="PDupltFxga8FYwjpE7t3UYZIBKRdpky4cpV8QpcF"

# Inicia la API con uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
