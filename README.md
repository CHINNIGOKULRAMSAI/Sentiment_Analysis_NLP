# 🚀 **FAKE NEWS DETECTION — NLP + FASTAPI + DOCKER + AZURE**

### *A Production-Ready Fake News Classifier deployed with Azure App Service*

<p align="center">
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Container-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Cloud-Azure-0089D6?style=for-the-badge&logo=microsoftazure&logoColor=white"/>
  <img src="https://img.shields.io/badge/Language-Python_3.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Model-ML_NLP-success?style=for-the-badge"/>
</p>

---

## 📌 **Live Demo**

🔹 **Frontend Web App:**
👉 [https://gokulsentimentweb.azurewebsites.net](https://gokulsentimentweb.azurewebsites.net)

🔹 **API Documentation (Swagger UI):**
👉 [https://gokulsentimentweb.azurewebsites.net/docs](https://gokulsentimentweb.azurewebsites.net/docs)

---

## ✨ **Overview**

This project is a real-time **Sentiment Analysis System** capable of predicting whether text is:

### ✅ POSITIVE

### ❌ NEGATIVE

The system uses:

* NLP preprocessing
* TF-IDF vectorization
* Logistic Regression / Linear SVC model
* FastAPI backend
* Docker containerization
* Azure ACR + Azure App Service deployment

The UI is clean, modern, and looks professional for production use.

---

## 🧠 **Features**

### 🌟 Core Capabilities

* Real-time sentiment prediction
* Confidence score output
* Robust NLP preprocessing

  * Lowercasing
  * Stopword removal
  * Lemmatization
* Model + Vectorizer loaded from pickle
* Custom prediction pipeline
* Beautiful frontend interface
* SEO-friendly design

---

### 🌐 Cloud & DevOps

* Fully Dockerized ML API
* Container pushed to Azure Container Registry (ACR)
* Auto-deployed to Azure Web App (Linux)
* `/health` endpoint for uptime monitoring
* `/docs` Swagger API documentation
* Cloud logs available via Azure Log Stream

---

## 📁 **Project Structure**

```
Sentiment_Analysis/
│
├── src/
│   ├── components/
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── predict_pipeline.py
│   │
│   ├── utils/
│   │   └── utils.py
│   │
│   ├── exception/
│   │   └── CustomException.py
│   │
│   ├── logger/
│       └── logging.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── artifacts/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 🏗️ **System Architecture**

```
User Browser (Frontend UI)
        |
        ▼
     FastAPI API (/predict, /health)
        |
        ▼
  ML Model (TF-IDF + Classifier)
        |
        ▼
    Docker Container
        |
        ▼
Azure Container Registry (ACR)
        |
        ▼
Azure App Service (Production)
```

---

## 🔧 **Tech Stack**

| Layer            | Technology                                      |
| ---------------- | ----------------------------------------------- |
| Frontend         | HTML, CSS, JavaScript                           |
| Backend API      | FastAPI                                         |
| ML Model         | Scikit-learn (TF-IDF + Logistic Regression/SVC) |
| Containerization | Docker                                          |
| Cloud Deployment | Azure ACR + Azure App Service                   |
| Monitoring       | Azure Log Stream                                |

---

## 🚀 **Local Development**

### 1️⃣ Clone Repository

```bash
git clone https://github.com/CHINNIGOKULRAMSAI/Sentiment_Analysis_NLP.git
cd Sentiment_Analysis_NLP
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run FastAPI App

```bash
uvicorn main:app --reload --port 8000
```

Open in browser:

* Frontend → [http://127.0.0.1:8000](http://127.0.0.1:8000)
* API Docs → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🐳 **Docker Setup**

### Build Image

```bash
docker build -t sentiment-app:v1 .
```

### Run Container

```bash
docker run -p 8000:8000 sentiment-app:v1
```

---

## 🌩️ **Azure Deployment (ACR + App Service)**

*(Deployment already completed — steps included for reference.)*

1. Create Resource Group
2. Create Container Registry
3. Build image locally
4. Push image to ACR
5. Create App Service Plan
6. Create Web App
7. Configure container
8. Restart & deploy

---

## 🎯 **Conclusion**

This project demonstrates **full ML deployment** with:

* NLP
* FastAPI
* Docker
* Azure Cloud

It is fully production-ready and ideal for **portfolio, research, or scalable SaaS projects**.

---
