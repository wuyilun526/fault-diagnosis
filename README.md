# RAG-based Fault Diagnosis System with Alibaba Cloud's Qwen Model

## Overview
This project implements an intelligent fault diagnosis system that combines **Retrieval-Augmented Generation (RAG)** technology with **Alibaba Cloud's Qwen large language model**. The system processes alerts containing descriptions, metrics, and log information, cross-references them against a historical fault knowledge base, and outputs diagnostic results with recommended actions.

## Key Features
- 🧠 **Multi-source analysis**: Simultaneously processes alert descriptions, metrics data, and log information
- 🔍 **Hybrid retrieval**: Combines BM25 keyword search with custom semantic vector retrieval
- 🤖 **LLM-enhanced reasoning**: Utilizes Alibaba's Qwen model for intelligent diagnosis
- 📚 **Lightweight knowledge base**: Custom embedding model without external dependencies
- ⚡ **Real-time response**: Average diagnostic response time < 3 seconds
- 🔧 **Extensible architecture**: Supports seamless switching between different LLM APIs

## Quick Start
### Prerequisites
1.Install Python 3.9+
2.Obtain Alibaba Cloud Qwen API Key

### Installation
```Bash
pip install -r requirements.txt
```

### Environment Setup
```Bash
# Linux/macOS
export ALI_API_KEY=your_api_key_here
# Windows
set ALI_API_KEY=your_api_key_here
```

### Running the System
```Bash
python fault_diagnosis.py
```

## Configuration
- Key configuration parameters in the code:
```python
CONFIG = {
    "ali_api_key": os.getenv("ALI_API_KEY"),
    "llm_model": "qwen-turbo",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 3,
}
```

## Custom Knowledge Base
- Modify the historical_data variable:
```python
historical_data = [
    {
        "symptom": "Database response latency increased",
        "category": "Database performance issue",
        "solution": "1. Check slow queries\n2. Optimize indexes\n3. Increase caching"
    },
    {
        "symptom": "Service memory usage continuously growing",
        "category": "Memory leak",
        "solution": "1. Generate heap dump\n2. Analyze memory objects\n3. Fix reference chains"
    }
]
```

## Demo
### Input
```python
Alert Description: Order service timeout rate exceeds 60%
Metrics Data: DB connection pool at 100% utilization
Log Information: Numerous 'Lock wait timeout' error logs
```

### Output
```python
1. Fault Category: Database Deadlock
2. Diagnosis Evidence:
   - 'Lock wait timeout' logs match historical cases
   - DB connection pool saturation consistent with deadlock
3. Immediate Actions:
   Check database lock wait chains
4. Notes:
   Optimize indexes during off-peak hours
```

## Application Scenarios
- Enterprise operations centers
- Cloud service monitoring
- Industrial IoT systems
- Application health monitoring

## Support
- wuyilun526@163.com