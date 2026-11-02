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
    && runitback nfl fetch-model --dest "$RUNITBACK_NFL_CKPT"

COPY --from=web-build /app/web/dist ./web/dist
# No NBA model is released yet: the NBA page explains how to train one. To ship one, copy a
# checkpoint to runs/nba/prod/model.pt in the image (e.g. COPY runs/nba/prod/model.pt ...).

EXPOSE 8080
CMD ["sh", "-c", "uvicorn runitback.server:app --host 0.0.0.0 --port ${PORT:-8080}"]