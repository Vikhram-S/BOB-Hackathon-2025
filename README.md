# 🛡️ Hybrid Identity Monitoring & Deepfake-Resistant Verification

**Bank of Baroda Hackathon Solution**

A comprehensive system for continuous identity monitoring and deepfake-resistant verification in hybrid cloud/on-premises environments.

## 🎯 Problem Statement

With the rise of AI-generated deepfakes, the integrity of digital identity verification processes like Video KYC (VKYC) is at risk. This solution addresses the need for continuous identity assurance in hybrid ecosystems (on-prem + cloud) while providing robust protection against deepfake attacks.

## ✨ Key Features

### 🔍 Advanced Deepfake Detection
- **Multi-layered Analysis**: Facial consistency, eye blinking patterns, head movement analysis
- **AI-Powered Detection**: Machine learning models for artifact detection
- **Real-time Processing**: Fast video analysis with configurable thresholds
- **High Accuracy**: Combines multiple detection techniques for robust results

### 🎥 Video KYC (Know Your Customer)
- **Liveness Verification**: Ensures the person in the video is alive and present
- **Face Quality Assessment**: Evaluates video quality for reliable verification
- **Document Consistency**: Cross-verification with uploaded identity documents
- **Anti-Deepfake Integration**: Built-in deepfake detection during KYC process

### 👁️ Continuous Identity Monitoring
- **Behavioral Analytics**: User and Entity Behavior Analytics (UEBA)
- **Anomaly Detection**: Machine learning-based anomaly detection
- **Risk Assessment**: Comprehensive risk scoring and alerting
- **Real-time Monitoring**: Continuous surveillance of user activities

### 🏗️ Hybrid Deployment
- **Flexible Architecture**: Supports on-premises, cloud, and hybrid deployments
- **Load Balancing**: Intelligent request routing based on workload
- **Data Synchronization**: Seamless data sync between on-prem and cloud
- **Scalability**: Auto-scaling capabilities for high-volume processing

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- 8GB+ RAM (for AI model processing)
- GPU recommended (for faster deepfake detection)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd BOB_Hackathon
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Initialize database**
```bash
python -c "from models.database import create_tables; create_tables()"
```

5. **Run the application**
```bash
# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000

# In another terminal, start the dashboard
streamlit run streamlit_dashboard.py
```

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
cd deployment
docker-compose up -d
```

2. **Access the services**
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Web Interface: http://localhost:8000/dashboard

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Streamlit     │    │   Mobile App    │
│   (Dashboard)   │    │   Dashboard     │    │   (Future)       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     FastAPI Gateway       │
                    │   (Request Routing)       │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                       │                         │
┌───────▼────────┐    ┌─────────▼─────────┐    ┌─────────▼─────────┐
│  On-Premises   │    │   Hybrid Manager   │    │   Cloud Service   │
│  Processing    │    │   (Load Balancer)  │    │   (Scalable)      │
└────────────────┘    └───────────────────┘    └───────────────────┘
        │                       │                         │
        └───────────────────────┼─────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Core Services       │
                    │ ┌─────────────────┐   │
                    │ │ Deepfake        │   │
                    │ │ Detector        │   │
                    │ └─────────────────┘   │
                    │ ┌─────────────────┐   │
                    │ │ Video KYC       │   │
                    │ │ Verifier        │   │
                    │ └─────────────────┘   │
                    │ ┌─────────────────┐   │
                    │ │ Identity       │   │
                    │ │ Monitor        │   │
                    │ └─────────────────┘   │
                    └───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Data Layer          │
                    │ ┌─────────────────┐   │
                    │ │ PostgreSQL      │   │
                    │ │ (Primary DB)    │   │
                    │ └─────────────────┘   │
                    │ ┌─────────────────┐   │
                    │ │ Redis Cache     │   │
                    │ │ (Sessions)      │   │
                    │ └─────────────────┘   │
                    └───────────────────────┘
```

## 🔧 Configuration

### Environment Variables

```bash
# Application Settings
APP_NAME="Hybrid Identity Monitoring System"
DEBUG=false
SECRET_KEY="your-secret-key-change-in-production"

# Database
DATABASE_URL="sqlite:///./identity_monitoring.db"  # or PostgreSQL URL

# Redis
REDIS_URL="redis://localhost:6379"

# Deepfake Detection
DEEPFAKE_MODEL_PATH="./models/deepfake_detector.h5"
CONFIDENCE_THRESHOLD=0.8

# Deployment
DEPLOYMENT_MODE="hybrid"  # "on-prem", "cloud", "hybrid"
ON_PREM_ENDPOINT="http://localhost:8000"
CLOUD_ENDPOINT="https://cloud.identity-monitoring.com"

# Monitoring
MONITORING_INTERVAL_SECONDS=60
ALERT_THRESHOLD_ANOMALY_SCORE=0.7
```

### Model Configuration

The system supports multiple deepfake detection models:

1. **Built-in Models**: MediaPipe-based facial analysis
2. **Custom Models**: TensorFlow/Keras models
3. **Hybrid Approach**: Combines multiple detection methods

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

### User Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/users/register` | POST | Register new user |
| `/api/v1/users/{user_id}` | GET | Get user information |

### Video KYC

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/kyc/video` | POST | Process Video KYC verification |

### Deepfake Detection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/deepfake/detect` | POST | Detect deepfake in video |

### Identity Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/monitoring/check` | POST | Check identity monitoring |
| `/api/v1/monitoring/dashboard/{user_id}` | GET | Get monitoring dashboard |
| `/api/v1/monitoring/start/{user_id}` | POST | Start continuous monitoring |
| `/api/v1/monitoring/stop/{user_id}` | POST | Stop continuous monitoring |

### Alerts

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/alerts` | GET | Get system alerts |
| `/api/v1/alerts/{alert_id}/resolve` | POST | Resolve alert |

### Deployment

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/deployment/status` | GET | Get deployment status |

## 🎮 Usage Examples

### 1. Video KYC Verification

```python
import requests

# Upload video for KYC
files = {
    'video_file': open('kyc_video.mp4', 'rb'),
    'document_image': open('id_document.jpg', 'rb')
}
data = {'user_id': 'user_001'}

response = requests.post(
    'http://localhost:8000/api/v1/kyc/video',
    files=files,
    data=data
)

result = response.json()
print(f"Verification Status: {result['verification_status']}")
print(f"Confidence Score: {result['confidence_score']}")
```

### 2. Deepfake Detection

```python
# Check video for deepfake
files = {'video_file': open('suspicious_video.mp4', 'rb')}

response = requests.post(
    'http://localhost:8000/api/v1/deepfake/detect',
    files=files
)

result = response.json()
if result['is_deepfake']:
    print(f"⚠️ Deepfake detected! Confidence: {result['confidence']}")
else:
    print(f"✅ Real video detected. Confidence: {result['confidence']}")
```

### 3. Identity Monitoring

```python
# Start continuous monitoring
response = requests.post(
    'http://localhost:8000/api/v1/monitoring/start/user_001'
)

# Check monitoring status
response = requests.get(
    'http://localhost:8000/api/v1/monitoring/dashboard/user_001'
)

dashboard_data = response.json()
print(f"Anomaly Score: {dashboard_data['activity_stats']['avg_anomaly_score']}")
```

## 🚀 Deployment Options

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn main:app --reload

# Run dashboard
streamlit run streamlit_dashboard.py
```

### 2. Docker Deployment
```bash
# Build and run
docker-compose up -d

# Scale services
docker-compose up -d --scale identity-api=3
```

### 3. Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/identity-monitoring.yaml

# Check deployment status
kubectl get pods -n identity-monitoring
```

### 4. Hybrid Cloud Deployment

The system supports flexible deployment configurations:

- **On-Premises Only**: All processing on local infrastructure
- **Cloud Only**: Fully cloud-based deployment
- **Hybrid**: Intelligent routing between on-prem and cloud

## 📊 Monitoring & Observability

### System Metrics
- **Performance**: Response times, throughput, error rates
- **Security**: Deepfake detection rates, anomaly scores
- **Infrastructure**: CPU, memory, disk usage
- **Business**: Verification success rates, user activity

### Alerting
- **Critical**: Deepfake detected, security breach
- **High**: High risk user, system errors
- **Medium**: Anomalous behavior, low confidence
- **Low**: System notifications, status updates

### Dashboards
1. **Web Dashboard**: HTML/CSS/JavaScript interface
2. **Streamlit Dashboard**: Interactive Python dashboard
3. **API Endpoints**: Programmatic access to metrics

## 🔒 Security Features

### Data Protection
- **Encryption**: All data encrypted in transit and at rest
- **Access Control**: Role-based access control (RBAC)
- **Audit Logging**: Comprehensive audit trails
- **Data Privacy**: GDPR/CCPA compliant data handling

### Anti-Deepfake Measures
- **Multi-layered Detection**: Facial, behavioral, and technical analysis
- **Real-time Processing**: Immediate deepfake detection
- **Continuous Learning**: Model updates and improvements
- **False Positive Reduction**: Advanced filtering and validation

### Identity Assurance
- **Continuous Monitoring**: Real-time identity verification
- **Risk Assessment**: Dynamic risk scoring
- **Behavioral Analysis**: User behavior pattern recognition
- **Anomaly Detection**: Unusual activity identification

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### Deepfake Detection Testing
```bash
# Test with sample videos
python tests/deepfake_test.py
```

## 📈 Performance Optimization

### Model Optimization
- **Quantization**: Reduce model size and inference time
- **Pruning**: Remove unnecessary model parameters
- **Distillation**: Transfer knowledge to smaller models
- **Caching**: Cache model predictions for repeated inputs

### System Optimization
- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Caching**: Redis-based caching for frequent queries
- **Load Balancing**: Distribute load across instances

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Team**
**Theme**: Hybrid Identity Monitoring & Deepfake-Resistant Verification  
- Vikhram S
- Kailash Karthikeyan M
- Gowtham P G

## 📞 Support

For support and questions:
- 📧 Email: vikhrams@saveetha.ac.in
- 🐛 Issues: [GitHub Issues]

## 🙏 Acknowledgments

- Bank of Baroda for organizing this hackathon
- Open source community for the amazing tools and libraries
- AI/ML researchers for advancing deepfake detection techniques
- Security experts for identity verification best practices

---

**Built with ❤️ for the Bank of Baroda Hackathon by Vikhram S and Team**
