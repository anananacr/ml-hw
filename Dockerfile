FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN pip install uv

RUN uv sync

COPY "predict.py" "pipeline_v1.bin" ./

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
