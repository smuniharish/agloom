FROM node:24-alpine AS build
WORKDIR /app
ARG VITE_OIDC_URL
ARG VITE_OIDC_REALM=agloom
ARG VITE_OIDC_CLIENT_ID=evidence-lab
ENV VITE_OIDC_URL=$VITE_OIDC_URL VITE_OIDC_REALM=$VITE_OIDC_REALM VITE_OIDC_CLIENT_ID=$VITE_OIDC_CLIENT_ID
COPY examples/continuous_evidence_lab/frontend/package*.json ./
RUN npm ci
COPY examples/continuous_evidence_lab/frontend/ ./
RUN npm run build
FROM nginx:1.30-alpine
COPY examples/continuous_evidence_lab/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
