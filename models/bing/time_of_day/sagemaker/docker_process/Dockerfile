FROM continuumio/anaconda3:latest
COPY . /app
RUN ls /app
RUN ls /app/*
RUN pip install -r app/datascience-utils/requirements.txt && \
    pip install app/datascience-utils
ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3"]
