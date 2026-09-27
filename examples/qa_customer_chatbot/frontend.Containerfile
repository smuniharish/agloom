FROM node:22-alpine AS build

WORKDIR /app
COPY examples/qa_customer_chatbot/frontend/package*.json ./
RUN npm ci
COPY examples/qa_customer_chatbot/frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY examples/qa_customer_chatbot/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
