FROM goldengrapegentleman/unsloth-rocm:2026.1.4-rocm7.1-gfx1100

# Install as root for /opt/venv access
USER root

# Training dependencies not in the base image
RUN pip install --no-cache-dir liger-kernel gguf

# Switch back to non-root user
USER unsloth

WORKDIR /workspace
