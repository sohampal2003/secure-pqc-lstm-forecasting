#!/bin/bash

# Time Series Forecasting + PQC Deployment Script
# This script automates the deployment of the entire system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="time-series-forecasting-pqc"
DOCKER_IMAGE="time-series-forecasting-pqc"
DOCKER_TAG="latest"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data
    mkdir -p models
    mkdir -p logs
    mkdir -p ssl
    mkdir -p grafana/provisioning/datasources
    mkdir -p grafana/provisioning/dashboards
    
    print_success "Directories created"
}

# Function to generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    print_status "Generating SSL certificates..."
    
    if [ ! -f ssl/cert.pem ] || [ ! -f ssl/key.pem ]; then
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        print_success "SSL certificates generated"
    else
        print_status "SSL certificates already exist"
    fi
}

# Function to create Nginx configuration
create_nginx_config() {
    print_status "Creating Nginx configuration..."
    
    cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream streamlit {
        server streamlit-dashboard:8501;
    }
    
    upstream grafana {
        server grafana:3000;
    }
    
    upstream prometheus {
        server prometheus:9090;
    }
    
    upstream flower {
        server flower:5555;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=streamlit:10m rate=30r/s;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    server {
        listen 80;
        server_name localhost;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name localhost;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # Streamlit Dashboard
        location / {
            limit_req zone=streamlit burst=20 nodelay;
            proxy_pass http://streamlit;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_redirect off;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # Grafana
        location /grafana/ {
            proxy_pass http://grafana/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Prometheus
        location /prometheus/ {
            proxy_pass http://prometheus/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Flower (Celery monitoring)
        location /flower/ {
            proxy_pass http://flower/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF
    
    print_success "Nginx configuration created"
}

# Function to create Prometheus configuration
create_prometheus_config() {
    print_status "Creating Prometheus configuration..."
    
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'streamlit'
    static_configs:
      - targets: ['streamlit-dashboard:8501']
    metrics_path: /metrics

  - job_name: 'celery'
    static_configs:
      - targets: ['celery-worker:5555']
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    metrics_path: /metrics
EOF
    
    print_success "Prometheus configuration created"
}

# Function to create Grafana datasource configuration
create_grafana_datasources() {
    print_status "Creating Grafana datasource configuration..."
    
    cat > grafana/provisioning/datasources/datasource.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    
  - name: PostgreSQL
    type: postgres
    access: proxy
    url: postgres:5432
    database: forecasting_db
    user: forecasting_user
    secureJsonData:
      password: forecasting_password
    jsonData:
      sslmode: disable
      maxOpenConns: 100
      maxIdleConns: 100
      connMaxLifetime: 14400
EOF
    
    print_success "Grafana datasource configuration created"
}

# Function to create Grafana dashboard configuration
create_grafana_dashboards() {
    print_status "Creating Grafana dashboard configuration..."
    
    cat > grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    print_success "Grafana dashboard configuration created"
}

# Function to build and start services
deploy_services() {
    print_status "Building and starting services..."
    
    # Build the main application image
    print_status "Building Docker image..."
    docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
    
    # Start all services
    print_status "Starting services with Docker Compose..."
    docker-compose up -d
    
    print_success "Services deployed successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for Streamlit
    print_status "Waiting for Streamlit dashboard..."
    until curl -f http://localhost:8501 >/dev/null 2>&1; do
        sleep 5
    done
    print_success "Streamlit dashboard is ready"
    
    # Wait for Grafana
    print_status "Waiting for Grafana..."
    until curl -f http://localhost:3000 >/dev/null 2>&1; do
        sleep 5
    done
    print_success "Grafana is ready"
    
    # Wait for Prometheus
    print_status "Waiting for Prometheus..."
    until curl -f http://localhost:9090 >/dev/null 2>&1; do
        sleep 5
    done
    print_success "Prometheus is ready"
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    echo ""
    
    # Docker containers status
    docker-compose ps
    
    echo ""
    print_status "Service URLs:"
    echo "  Streamlit Dashboard: https://localhost"
    echo "  Grafana: https://localhost/grafana/"
    echo "  Prometheus: https://localhost/prometheus/"
    echo "  Flower (Celery): https://localhost/flower/"
    echo ""
    print_status "Default credentials:"
    echo "  Grafana: admin/admin"
    echo ""
}

# Function to show logs
show_logs() {
    print_status "Showing recent logs..."
    docker-compose logs --tail=50
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose down
    print_success "Services stopped"
}

# Function to clean up everything
cleanup() {
    print_warning "This will remove all containers, volumes, and data. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --remove-orphans
        docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} 2>/dev/null || true
        rm -rf data models logs
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy      Deploy all services (default)"
    echo "  start       Start existing services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show service logs"
    echo "  cleanup     Remove all services and data"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy    # Deploy everything"
    echo "  $0 status    # Check service status"
    echo "  $0 logs      # View logs"
}

# Main script logic
main() {
    case "${1:-deploy}" in
        "deploy")
            check_requirements
            create_directories
            generate_ssl_certificates
            create_nginx_config
            create_prometheus_config
            create_grafana_datasources
            create_grafana_dashboards
            deploy_services
            wait_for_services
            show_status
            ;;
        "start")
            docker-compose start
            print_success "Services started"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            docker-compose up -d
            print_success "Services restarted"
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
