FROM python:3.9-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
	build-essential \
	libfreetype6-dev \
	libpng-dev \
 && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Tested_Monolithic_GA.py .

ENTRYPOINT ["python", "Tested_Monolithic_GA.py"]
