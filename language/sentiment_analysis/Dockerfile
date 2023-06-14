FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt
COPY ./app/model /app/app/model

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app
