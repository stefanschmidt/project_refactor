FROM frolvlad/alpine-miniconda3

EXPOSE 5001/tcp
WORKDIR /app

RUN conda install flask pandas numpy scikit-learn imbalanced-learn plotly

# because of kaleido binary executable
RUN pip install kaleido
RUN conda install -c plotly plotly-orca

ENV FLASK_ENV=development
ENV FLASK_APP=app.py

COPY data/telecom_users.pkl ./data/telecom_users.pkl
COPY clustering_label.py .
COPY templates .
COPY app.py .

CMD [ "python", "./app.py" ]

