FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 

RUN python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words

COPY . . 

EXPOSE 5000 

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]