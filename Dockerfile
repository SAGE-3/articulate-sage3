FROM python:3.9

# RUN apt-get update && apt-get install -y iputils-ping

WORKDIR /app

COPY src /app

RUN pip install --no-cache-dir -r requirements.txt