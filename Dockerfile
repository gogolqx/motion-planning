FROM python:3.7
COPY ./clustering ./clustering
COPY ./entrypoint.sh ./clustering/entrypoint.sh
COPY ./requirements.txt ./clustering/requirements.txt
WORKDIR ./clustering
RUN python3.7 -m pip install --no-cache-dir -r requirements.txt && chmod +x entrypoint.sh
