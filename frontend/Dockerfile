FROM node:20-alpine as build

WORKDIR /app

COPY /frontend/package.json /frontend/package-lock.json* ./

RUN npm install --frozen-lockfile --legacy-peer-deps

COPY . .

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
