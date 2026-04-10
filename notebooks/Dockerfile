# 1. Empezamos con una imagen base de Python (instrucción FROM)
FROM python:3.9-slim

# 2. Definimos nuestro directorio de trabajo en el contenedor (instrucción WORKDIR)
WORKDIR /app

# 3. Copiamos nuestro código y el archivo de requerimientos (instrucción COPY)
COPY requirements.txt .
COPY SparkML.ipynb .

# 4. Instalamos las dependencias necesarias (instrucción RUN)
RUN pip install -r requirements.txt

# 5. Qué comando ejecutará el contenedor al arrancar (instrucción CMD)
CMD ["python", "SparkML.ipynb"]
