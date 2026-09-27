FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /opt/caps

COPY requirements.txt pyproject.toml README.md LICENSE NOTICE ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN python -m pip install --no-deps -e .

RUN useradd --create-home --shell /bin/bash runner \
    && mkdir -p /opt/caps/reports/scratch \
    && chown -R runner:runner /opt/caps
USER runner

ENTRYPOINT ["python", "-m", "private_vascular_motion_generation.cli.verify"]
CMD ["--root", ".", "--scratch", "reports/scratch"]
