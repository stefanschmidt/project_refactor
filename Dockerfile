FROM python:3.12-alpine

RUN apk update && apk add build-base

EXPOSE 5001/tcp
WORKDIR /app

RUN pip install flask

ENV FLASK_ENV=development
ENV FLASK_APP=app.py

COPY clustering_label.py .
COPY static ./static
COPY templates ./templates
COPY app.py .

CMD [ "python", "./app.py" ]

