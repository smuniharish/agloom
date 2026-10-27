FROM node:24-alpine AS build

WORKDIR /app
COPY examples/rca_generator/frontend/package*.json ./
RUN npm ci
COPY examples/rca_generator/frontend/ ./
RUN npm run build

FROM nginx:1.30-alpine
COPY examples/rca_generator/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
