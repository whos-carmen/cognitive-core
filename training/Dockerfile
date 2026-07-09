FROM ghcr.io/unslothai/unsloth:2025.6.1-cuda

# Install as root for /opt/venv access
USER root

# Copy and install training dependencies
COPY requirements.txt /tmp/requirements.txt
RUN (uv pip install --system -r /tmp/requirements.txt 2>/dev/null || \
    pip install --no-cache-dir -r /tmp/requirements.txt) && \
    rm /tmp/requirements.txt

# Switch back to non-root user
USER unsloth

WORKDIR /workspace
