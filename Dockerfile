# Use an official Python 3.12.9 base image
FROM python:3.12.9-slim

# Set environment variables
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install uv

# Create a virtual environment using uv
RUN uv venv --python=3.12

# Install Python packages inside the virtual environment
RUN uv pip install \
    langchain_core \
    langchain_chroma \
    langchain_openai \
    langchain_huggingface \
    pymongo \
    gradio

# (Optional) Copy your project files into /app
# COPY . .

# Default command (bash shell for now)
CMD ["bash"]
