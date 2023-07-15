FROM python:3.10

ARG UID
ARG GID

WORKDIR /app

RUN groupadd --gid $GID myuser && useradd --no-create-home -u $UID --gid $GID myuser && chown -R myuser:myuser /app && mkdir /home/myuser && chown myuser:myuser /home/myuser

ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY requirements.txt enable_extensions.sh jupyter_extensions.txt .

RUN pip --no-cache-dir install -r requirements.txt

USER myuser

RUN jupyter-contrib nbextension install --user && ./enable_extensions.sh

# docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --tag ml_pipeline_img .
# docker run -p 8888:8888 -p 5000:5000 -p 8080:8080 --add-host=dockerhost:0.0.0.0 --user "$(id -u):$(id -g)" --name MLpipeline_con --rm -it -v $(pwd):/app ml_pipeline_img bash
# docker exec -it MLpipeline_con bash
# dvc repro --force learn_and_evaluation
# mlflow ui --host 0.0.0.0
# jupyter notebook --ip=0.0.0.0
# ./format_scripts.sh script
# pylint --max-line-length=88
