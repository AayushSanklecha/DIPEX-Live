# ─────────────────────────────────────────────────────────────────
# DIPEX Frontend — Dockerfile
# Multi-stage build for React/Vite SPA
# ─────────────────────────────────────────────────────────────────

# Stage 1: Build the React application
FROM node:20-alpine as builder

WORKDIR /app

# Install dependencies first for layer caching
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy the built assets from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
