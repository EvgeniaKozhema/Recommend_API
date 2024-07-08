FROM python:3.10.13

WORKDIR /usr/src/app/recommend_api

COPY poetry.lock pyproject.toml /usr/src/app/recommend_api/

RUN pip install poetry

RUN poetry install --no-dev

COPY . /usr/src/app/recommend_api/

CMD ["poetry", "run","python", "-m", "src.ab_app"]
