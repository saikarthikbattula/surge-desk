# SurgeDesk container for IBM LinuxONE (s390x).
# Base: IBM's LinuxONE Community Cloud JupyterLab image, which already has PyTorch, XGBoost,
# scikit-learn, transformers, and llama-cpp-python built with zDNN/NNPA acceleration.
# Our image adds the SurgeDesk code. Data, models, and outputs are mounted at runtime
# (the Granite file is ~2.7 GB, so it is not baked into the image).
#
# Build (VM, repo root):  docker build -t surgedesk:latest .
# Run (VM):               see handoffs/person1_vm_runner.md section 6
FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest

LABEL org.opencontainers.image.title="SurgeDesk" \
      org.opencontainers.image.description="Disaster report triage on IBM LinuxONE"

COPY --chown=1000:1000 . /home/jovyan/surgedesk
WORKDIR /home/jovyan/surgedesk

# Intentionally empty unless a human adds a line (see AGENTS.md rule 2).
RUN if [ -s requirements-extra.txt ]; then python -m pip install --no-cache-dir -r requirements-extra.txt; fi

# Code lives in the image; data/models/outputs live in the mounted repo folder on the host.
ENV PYTHONPATH=/home/jovyan/surgedesk \
    SURGEDESK_ROOT=/home/jovyan/shared/surgedesk \
    SURGEDESK_THREADS=2
