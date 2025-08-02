# Production Deployment Guide

This guide covers best practices and strategies for deploying Atlas in production environments, from single-server deployments to distributed cloud architectures.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Architectures](#deployment-architectures)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [API Gateway Setup](#api-gateway-setup)
7. [Security Considerations](#security-considerations)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting Production Issues](#troubleshooting-production-issues)

## Pre-Deployment Checklist

### Code Readiness

- [ ] All tests passing (unit, integration, performance)
- [ ] Code coverage > 80%
- [ ] No security vulnerabilities (run security scan)
- [ ] Documentation updated
- [ ] Performance benchmarks meet requirements
- [ ] Error handling for all edge cases
- [ ] Logging configured appropriately
- [ ] Configuration externalized

### Infrastructure Requirements

- [ ] Server specifications defined
- [ ] Database provisioned and tested
- [ ] Network architecture planned
- [ ] Load balancer configured
- [ ] SSL certificates obtained
- [ ] DNS entries configured
- [ ] Backup strategy defined
- [ ] Monitoring tools set up

### Operational Readiness

- [ ] Deployment procedures documented
- [ ] Rollback plan defined
- [ ] On-call rotation established
- [ ] Incident response procedures
- [ ] Performance baselines established
- [ ] Capacity planning completed
- [ ] SLAs defined and agreed

## Deployment Architectures

### Single Server Deployment

Simple deployment for small-scale usage:

```
┌─────────────────────────────────────┐
│           Load Balancer             │
│              (Nginx)                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│         Application Server          │
│      ┌─────────────────────┐        │
│      │  Optimizer Service  │        │
│      │    (Gunicorn)       │        │
│      └──────────┬──────────┘        │
│                 │                   │
│      ┌──────────┴──────────┐        │
│      │   Model Services    │        │
│      │    (Docker)         │        │
│      └──────────┬──────────┘        │
│                 │                   │
│      ┌──────────┴──────────┐        │
│      │     Database        │        │
│      │   (PostgreSQL)      │        │
│      └─────────────────────┘        │
└─────────────────────────────────────┘
```

### Microservices Architecture

Scalable deployment for enterprise usage:

```
┌─────────────────────────────────────────────────┐
│                 API Gateway                     │
│                (Kong/Traefik)                   │
└────────┬────────────┬──────────┬────────────────┘
         │            │          │
    ┌────┴────┐  ┌────┴────┐  ┌──┴──────┐
    │Optimizer│  │  Model  │  │ Results │
    │ Service │  │Registry │  │ Service │
    └────┬────┘  └────┬────┘  └──┬──────┘
         │            │          │
    ┌────┴────────────┴──────────┴────┐
    │         Message Queue           │
    │         (RabbitMQ/Kafka)        │
    └────────────┬────────────────────┘
                 │
    ┌────────────┴───────────────────┐
    │      Model Services Farm       │
    │  ┌──────┐ ┌──────┐ ┌──────┐    │
    │  │Model1│ │Model2│ │Model3│    │
    │  └──────┘ └──────┘ └──────┘    │
    └────────────────────────────────┘
```

## Docker Deployment

### Production Dockerfile

```dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim-bullseye as builder

# Build arguments
ARG VERSION
ARG BUILD_DATE

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 optimizer

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=optimizer:optimizer . .

# Final stage
FROM python:3.11-slim-bullseye

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /home/optimizer/.local /home/optimizer/.local
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 optimizer
USER optimizer

# Set environment
ENV PATH=/home/optimizer/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV OPTIMIZER_VERSION=${VERSION}

# Labels
LABEL version=${VERSION} \
      build-date=${BUILD_DATE} \
      description="Atlas Production Image"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
WORKDIR /app
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "optimizer_framework.server:app"]
```

### Docker Compose Production

```yaml
version: '3.8'

services:
  optimizer:
    image: atlas:${VERSION:-latest}
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://optimizer:${DB_PASSWORD}@db:5432/optimizer
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - WORKERS=${WORKERS:-4}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./config:/app/config:ro
      - model-cache:/app/models
    networks:
      - optimizer-network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  db:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=optimizer
      - POSTGRES_USER=optimizer
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - optimizer-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U optimizer"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - optimizer-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - optimizer
    networks:
      - optimizer-network

volumes:
  postgres-data:
  redis-data:
  model-cache:

networks:
  optimizer-network:
    driver: bridge
```

### Production Nginx Configuration

```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;

    # Security
    server_tokens off;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=optimize:10m rate=1r/s;

    # Upstream
    upstream optimizer_backend {
        least_conn;
        server optimizer:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # HTTPS redirect
    server {
        listen 80;
        server_name optimizer.example.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name optimizer.example.com;

        # SSL
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://optimizer_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 300s;  # Long timeout for optimization
        }

        # Optimization endpoint (stricter rate limit)
        location /api/optimize {
            limit_req zone=optimize burst=5 nodelay;
            
            proxy_pass http://optimizer_backend;
            # Same proxy settings as above
        }

        # Health check
        location /health {
            access_log off;
            proxy_pass http://optimizer_backend;
        }

        # Static files
        location /static/ {
            alias /app/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: atlas
  namespace: optimizer
  labels:
    app: optimizer
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: optimizer
  template:
    metadata:
      labels:
        app: optimizer
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: optimizer-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: optimizer
        image: atlas:1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: optimizer-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: optimizer-config
              key: redis-url
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: models
          mountPath: /app/models
      volumes:
      - name: config
        configMap:
          name: optimizer-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - optimizer
              topologyKey: kubernetes.io/hostname
```

### Service and Ingress

```yaml
# Service
apiVersion: v1
kind: Service
metadata:
  name: optimizer-service
  namespace: optimizer
  labels:
    app: optimizer
spec:
  type: ClusterIP
  selector:
    app: optimizer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http

---
# Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: optimizer-ingress
  namespace: optimizer
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "10"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - api.optimizer.example.com
    secretName: optimizer-tls
  rules:
  - host: api.optimizer.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: optimizer-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: optimizer-hpa
  namespace: optimizer
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: atlas
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## Cloud Deployments

### AWS Deployment

#### ECS Task Definition

```json
{
  "family": "atlas",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "optimizer",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/optimizer:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "AWS_REGION",
          "value": "us-east-1"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:optimizer/db-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/optimizer",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ],
  "taskRoleArn": "arn:aws:iam::123456789:role/optimizer-task-role",
  "executionRoleArn": "arn:aws:iam::123456789:role/optimizer-execution-role"
}
```

#### CloudFormation Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Atlas Infrastructure'

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true

  # Application Load Balancer
  ALB:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Type: application
      Scheme: internet-facing
      SecurityGroups:
        - !Ref ALBSecurityGroup
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2

  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub optimizer-${Environment}
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      DefaultCapacityProviderStrategy:
        - CapacityProvider: FARGATE
          Weight: 1
        - CapacityProvider: FARGATE_SPOT
          Weight: 3

  # RDS Database
  Database:
    Type: AWS::RDS::DBCluster
    Properties:
      Engine: aurora-postgresql
      EngineMode: serverless
      ScalingConfiguration:
        MinCapacity: 2
        MaxCapacity: 8
        AutoPause: true
        SecondsUntilAutoPause: 300

  # ElastiCache Redis
  RedisCluster:
    Type: AWS::ElastiCache::ReplicationGroup
    Properties:
      ReplicationGroupId: !Sub optimizer-${Environment}
      ReplicationGroupDescription: Cache for optimizer
      Engine: redis
      CacheNodeType: cache.t3.micro
      NumCacheClusters: 2
      AutomaticFailoverEnabled: true
```

### Google Cloud Deployment

```yaml
# Cloud Run Service
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: atlas
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "2"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      serviceAccountName: optimizer-sa@project.iam.gserviceaccount.com
      containers:
      - image: gcr.io/project/optimizer:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-url
              key: latest
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
          initialDelaySeconds: 30
          periodSeconds: 30
```

### Azure Deployment

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "environment": {
      "type": "string",
      "defaultValue": "production"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-09-01",
      "name": "[concat('optimizer-', parameters('environment'))]",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "optimizer",
            "properties": {
              "image": "optimizer.azurecr.io/optimizer:latest",
              "ports": [
                {
                  "port": 8000,
                  "protocol": "TCP"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 1,
                  "memoryInGB": 2
                },
                "limits": {
                  "cpu": 2,
                  "memoryInGB": 4
                }
              },
              "environmentVariables": [
                {
                  "name": "DATABASE_URL",
                  "secureValue": "[parameters('databaseUrl')]"
                }
              ]
            }
          }
        ],
        "osType": "Linux",
        "restartPolicy": "Always"
      }
    }
  ]
}
```

## API Gateway Setup

### Kong Configuration

```yaml
# kong.yaml
_format_version: "2.1"

services:
  - name: optimizer-service
    url: http://optimizer:8000
    retries: 3
    connect_timeout: 60000
    write_timeout: 300000
    read_timeout: 300000

routes:
  - name: optimizer-route
    service: optimizer-service
    paths:
      - /api
    strip_path: false

plugins:
  - name: rate-limiting
    service: optimizer-service
    config:
      minute: 60
      hour: 1000
      policy: local

  - name: key-auth
    service: optimizer-service

  - name: cors
    service: optimizer-service
    config:
      origins:
        - https://app.example.com
      methods:
        - GET
        - POST
        - PUT
        - DELETE
      headers:
        - Accept
        - Content-Type
        - Authorization
      credentials: true

  - name: prometheus
    config:
      per_consumer: true

consumers:
  - username: web-app
    keyauth_credentials:
      - key: ${WEB_APP_API_KEY}

  - username: mobile-app
    keyauth_credentials:
      - key: ${MOBILE_APP_API_KEY}
```

## Security Considerations

### Security Checklist

- [ ] All secrets in environment variables or secret management
- [ ] TLS/SSL enabled for all connections
- [ ] API authentication implemented (OAuth2/JWT)
- [ ] Rate limiting configured
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention
- [ ] XSS protection headers
- [ ] CORS properly configured
- [ ] Container running as non-root user
- [ ] Network policies restricting traffic
- [ ] Regular security scanning
- [ ] Audit logging enabled

### Security Configuration

```python
# security.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import jwt
import time

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# JWT Authentication
security = HTTPBearer()

class SecurityMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app
        app.state.limiter = limiter
        app.add_exception_handler(429, _rate_limit_exceeded_handler)
        
    async def verify_token(self, credentials: HTTPAuthorizationCredentials):
        token = credentials.credentials
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            if payload["exp"] < time.time():
                raise HTTPException(status_code=401, detail="Token expired")
            return payload
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response
```

## Monitoring and Observability

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Define metrics
optimization_requests = Counter(
    'optimization_requests_total',
    'Total optimization requests',
    ['method', 'status']
)

optimization_duration = Histogram(
    'optimization_duration_seconds',
    'Optimization request duration',
    ['method']
)

active_optimizations = Gauge(
    'active_optimizations',
    'Number of active optimizations'
)

model_prediction_time = Histogram(
    'model_prediction_seconds',
    'Model prediction time',
    ['model_name']
)

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

# Decorator for tracking
def track_optimization(method: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            active_optimizations.inc()
            
            try:
                result = await func(*args, **kwargs)
                optimization_requests.labels(method=method, status='success').inc()
                return result
            except Exception as e:
                optimization_requests.labels(method=method, status='error').inc()
                raise
            finally:
                active_optimizations.dec()
                duration = time.time() - start_time
                optimization_duration.labels(method=method).observe(duration)
        
        return wrapper
    return decorator
```

### Logging Configuration

```python
# logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

def setup_logging(level="INFO"):
    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger"
        }
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Specific loggers
    logging.getLogger("optimizer_framework").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    return root_logger

# Structured logging
class StructuredLogger:
    def __init__(self, logger):
        self.logger = logger
    
    def log_optimization(self, event, **kwargs):
        self.logger.info(
            event,
            extra={
                "event_type": "optimization",
                "timestamp": time.time(),
                **kwargs
            }
        )
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Atlas Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(optimization_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, optimization_duration_seconds)"
          }
        ]
      },
      {
        "title": "Active Optimizations",
        "targets": [
          {
            "expr": "active_optimizations"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(optimization_requests_total{status='error'}[5m])"
          }
        ]
      }
    ]
  }
}
```

## Backup and Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Configuration
BACKUP_DIR="/backups"
S3_BUCKET="s3://optimizer-backups"
RETENTION_DAYS=30

# Database backup
echo "Backing up database..."
pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/db-$(date +%Y%m%d-%H%M%S).sql.gz

# Model backup
echo "Backing up models..."
tar -czf $BACKUP_DIR/models-$(date +%Y%m%d-%H%M%S).tar.gz /app/models

# Configuration backup
echo "Backing up configuration..."
tar -czf $BACKUP_DIR/config-$(date +%Y%m%d-%H%M%S).tar.gz /app/config

# Upload to S3
echo "Uploading to S3..."
aws s3 sync $BACKUP_DIR $S3_BUCKET --exclude "*" --include "*.gz"

# Clean old backups
echo "Cleaning old backups..."
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete
aws s3 ls $S3_BUCKET | awk '{print $4}' | xargs -I {} bash -c \
  'if [ $(date -d "now - 30 days" +%s) -gt $(date -d "{}" +%s) ]; then aws s3 rm $S3_BUCKET/{}; fi'

echo "Backup complete!"
```

### Recovery Procedures

```bash
#!/bin/bash
# restore.sh

# Configuration
BACKUP_DATE=$1
S3_BUCKET="s3://optimizer-backups"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: ./restore.sh YYYYMMDD"
    exit 1
fi

# Download backups
echo "Downloading backups from $BACKUP_DATE..."
aws s3 cp $S3_BUCKET/db-$BACKUP_DATE.sql.gz /tmp/
aws s3 cp $S3_BUCKET/models-$BACKUP_DATE.tar.gz /tmp/
aws s3 cp $S3_BUCKET/config-$BACKUP_DATE.tar.gz /tmp/

# Restore database
echo "Restoring database..."
gunzip -c /tmp/db-$BACKUP_DATE.sql.gz | psql $DATABASE_URL

# Restore models
echo "Restoring models..."
tar -xzf /tmp/models-$BACKUP_DATE.tar.gz -C /

# Restore configuration
echo "Restoring configuration..."
tar -xzf /tmp/config-$BACKUP_DATE.tar.gz -C /

echo "Restore complete!"
```

## Troubleshooting Production Issues

### Common Issues and Solutions

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n optimizer

# Get heap dump
kubectl exec -it optimizer-pod -- jcmd 1 GC.heap_dump /tmp/heap.hprof
kubectl cp optimizer-pod:/tmp/heap.hprof ./heap.hprof

# Analyze with profiler
```

#### Slow Response Times

```python
# Add detailed timing
import time
from functools import wraps

def timing_middleware(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        timings = {}
        
        # Model prediction timing
        start = time.time()
        result = await func(*args, **kwargs)
        timings['total'] = time.time() - start
        
        # Log slow requests
        if timings['total'] > 5.0:
            logger.warning(f"Slow request: {timings}")
        
        return result
    return wrapper
```

#### Database Connection Issues

```python
# Connection pool monitoring
from sqlalchemy import create_engine, pool
import logging

engine = create_engine(
    DATABASE_URL,
    poolclass=pool.QueuePool,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Log pool status
@app.on_event("startup")
async def log_pool_status():
    while True:
        await asyncio.sleep(60)
        logger.info(f"DB Pool: {engine.pool.status()}")
```

### Production Debugging

```python
# debug_endpoints.py
from fastapi import APIRouter, Depends
from typing import Dict
import psutil
import gc

debug_router = APIRouter(prefix="/debug", tags=["debug"])

@debug_router.get("/status")
async def system_status() -> Dict:
    """Get system status (only in debug mode)."""
    process = psutil.Process()
    
    return {
        "memory": {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "percent": process.memory_percent()
        },
        "cpu": {
            "percent": process.cpu_percent(interval=1),
            "num_threads": process.num_threads()
        },
        "connections": len(process.connections()),
        "open_files": len(process.open_files()),
        "gc_stats": gc.get_stats()
    }

# Only enable in debug mode
if os.getenv("DEBUG_MODE") == "true":
    app.include_router(debug_router)
```

## Post-Deployment

### Health Checks

```python
# health.py
from fastapi import APIRouter, Response, status
from sqlalchemy import text
import redis

health_router = APIRouter()

@health_router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}

@health_router.get("/ready")
async def readiness_check(response: Response):
    """Detailed readiness check."""
    checks = {
        "database": False,
        "redis": False,
        "models": False
    }
    
    try:
        # Check database
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except:
        pass
    
    try:
        # Check Redis
        redis_client.ping()
        checks["redis"] = True
    except:
        pass
    
    try:
        # Check models loaded
        if len(model_registry.list_models()) > 0:
            checks["models"] = True
    except:
        pass
    
    # Set status code
    if not all(checks.values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return {
        "ready": all(checks.values()),
        "checks": checks
    }
```

### Smoke Tests

```python
# smoke_tests.py
import requests
import time

def run_smoke_tests(base_url):
    """Run smoke tests after deployment."""
    tests = []
    
    # Test 1: Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        tests.append({
            "test": "health_check",
            "passed": r.status_code == 200
        })
    except:
        tests.append({"test": "health_check", "passed": False})
    
    # Test 2: API accessible
    try:
        r = requests.get(f"{base_url}/api/v1/models", timeout=5)
        tests.append({
            "test": "api_accessible",
            "passed": r.status_code in [200, 401]  # 401 if auth required
        })
    except:
        tests.append({"test": "api_accessible", "passed": False})
    
    # Test 3: Simple optimization
    try:
        r = requests.post(
            f"{base_url}/api/v1/optimize",
            json={
                "budget": {"tv": 100000, "digital": 200000},
                "constraints": {"total": 300000}
            },
            timeout=30
        )
        tests.append({
            "test": "optimization",
            "passed": r.status_code in [200, 201]
        })
    except:
        tests.append({"test": "optimization", "passed": False})
    
    # Report results
    passed = sum(1 for t in tests if t["passed"])
    print(f"Smoke Tests: {passed}/{len(tests)} passed")
    
    for test in tests:
        status = "✓" if test["passed"] else "✗"
        print(f"  {status} {test['test']}")
    
    return all(t["passed"] for t in tests)

if __name__ == "__main__":
    success = run_smoke_tests("https://api.optimizer.example.com")
    exit(0 if success else 1)
```

## Maintenance

### Rolling Updates

```bash
#!/bin/bash
# rolling_update.sh

NEW_VERSION=$1
NAMESPACE="optimizer"
DEPLOYMENT="atlas"

echo "Starting rolling update to version $NEW_VERSION..."

# Update image
kubectl set image deployment/$DEPLOYMENT optimizer=atlas:$NEW_VERSION -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

# Run smoke tests
python smoke_tests.py

if [ $? -eq 0 ]; then
    echo "Rolling update successful!"
else
    echo "Smoke tests failed, rolling back..."
    kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE
    exit 1
fi
```

### Maintenance Mode

```python
# maintenance.py
from fastapi import Request, Response
import json

MAINTENANCE_MODE = False

@app.middleware("http")
async def maintenance_middleware(request: Request, call_next):
    if MAINTENANCE_MODE and not request.url.path.startswith("/health"):
        return Response(
            content=json.dumps({
                "error": "Service temporarily unavailable for maintenance",
                "retry_after": 3600
            }),
            status_code=503,
            headers={"Retry-After": "3600"},
            media_type="application/json"
        )
    
    return await call_next(request)
```

Remember: Always test deployment procedures in staging before applying to production!