FROM node:20-alpine as build

WORKDIR /app

COPY package.json package-lock.json* ./

RUN npm install --frozen-lockfile --legacy-peer-deps

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
