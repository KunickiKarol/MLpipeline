FROM python:3.10

ARG UID
ARG GID
# docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --tag pdiow-python .
# dvc repro --force learn_and_evaluation
# mlflow ui --host 0.0.0.0

# docker run -p 8888:8888 -p 5000:5000 -p 8080:8080 --add-host=dockerhost:0.0.0.0 --user "$(id -u):$(id -g)" --name lab3 --rm -it -v $(pwd):/app pdiow-python bash
# docker exec -it lab3 bash
WORKDIR /app

RUN groupadd --gid $GID myuser && useradd -u $UID --gid $GID -ms /bin/bash myuser && chown -R myuser /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

USER myuser


