FROM ghcr.io/unslothai/unsloth:latest

USER root

COPY requirements.txt /tmp/requirements.txt
RUN (uv pip install --system -r /tmp/requirements.txt 2>/dev/null || \
    pip install --no-cache-dir -r /tmp/requirements.txt) && \
    rm /tmp/requirements.txt

USER unsloth
WORKDIR /workspace
