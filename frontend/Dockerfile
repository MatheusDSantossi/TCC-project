FROM node:20-alpine as build

WORKDIR /app

# allow injecting REACT_APP_API_URL at build time
ARG REACT_APP_API_URL=""
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

COPY package.json package-lock.json* ./

RUN npm ci --frozen-lockfile --legacy-peer-deps

COPY . .

# Verify file existence in container
RUN ls -la && \
    echo "Checking index.html:" && \
    [ -f index.html ] && echo "✔️ index.html exists" || echo "❌ index.html missing"

# Add after COPY . . command
RUN pwd && \
    ls -la && \
    ls -la src/ && \
    ls -la public/


RUN npm run build

FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html
COPY ./nginx/nginx.conf /etc/nginx/conf.d/default.conf
