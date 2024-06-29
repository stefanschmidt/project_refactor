FROM frolvlad/alpine-miniconda3

EXPOSE 5001/tcp
WORKDIR /app

# plotly-orca doesn't work
RUN conda install -c conda-forge flask pandas numpy scikit-learn imbalanced-learn plotly python-kaleido

# because of kaleido binary executable
# RUN pip install kaleido
# or try python-kaleido from conda-forge

ENV FLASK_ENV=development
ENV FLASK_APP=app.py

COPY data/telecom_users.pkl ./data/telecom_users.pkl
COPY clustering_label.py .
COPY templates .
COPY app.py .

CMD [ "python", "./app.py" ]

