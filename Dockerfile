FROM ghcr.io/unslothai/unsloth:latest-cuda

# Install as root for /opt/venv access
USER root

# Training dependencies not in the base image
RUN pip install --no-cache-dir liger-kernel gguf

# Switch back to non-root user
USER unsloth

WORKDIR /workspace
