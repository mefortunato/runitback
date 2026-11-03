FROM node:22-bookworm-slim AS web-build

WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUNITBACK_NFL_CKPT=/app/runs/nfl/prod/model.pt \
    RUNITBACK_NBA_CKPT=/app/runs/nba/prod/model.pt \
    RUNITBACK_WEB=/app/web/dist

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && runitback nfl fetch-model --dest "$RUNITBACK_NFL_CKPT" \
    && runitback nba fetch-model --dest "$RUNITBACK_NBA_CKPT" --optional

COPY --from=web-build /app/web/dist ./web/dist
# The NBA model is the latest release in src/runitback/nba/release.py; until one is listed there
# the NBA page explains how to train one.

EXPOSE 8080
CMD ["sh", "-c", "uvicorn runitback.server:app --host 0.0.0.0 --port ${PORT:-8080}"]