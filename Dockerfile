# Match the project venv (Python 3.10). The patch level floats with the slim tag.
# Retrieval pins match the stack that built artifacts/pubmedqa-bge-small-v1.
# torch is 2.14.1 from the CPU index; the local eval stack is 2.14.1+cu126.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_CACHE=/home/user/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface/hub \
    RAG_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        ".[api,ui]" \
        "numpy==2.2.6" \
        "faiss-cpu==1.15.1" \
        "transformers==5.20.2" \
        "huggingface_hub==2.4.0" \
    && python -c "import faiss, huggingface_hub, numpy, torch, transformers; version=torch.__version__; assert version.startswith('2.14.1') and '+cpu' in version, version; assert numpy.__version__=='2.2.6'; assert faiss.__version__=='1.15.1'; assert transformers.__version__=='5.20.2'; assert huggingface_hub.__version__=='2.4.0'"

COPY artifacts/pubmedqa-bge-small-v1 ./artifacts/pubmedqa-bge-small-v1

RUN chown -R user:user /app /home/user
USER user

# Same revision the manifest records, so runtime from_pretrained hits this cache.
RUN python -c "import json; from pathlib import Path; from transformers import AutoModel, AutoTokenizer; manifest=json.loads(Path('artifacts/pubmedqa-bge-small-v1/manifest.json').read_text(encoding='utf-8')); model=manifest['model']; AutoTokenizer.from_pretrained(model['id'], revision=model['requested_revision']); AutoModel.from_pretrained(model['id'], revision=model['requested_revision'])"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
