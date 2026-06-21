# 🚦 Event-Driven Congestion Intelligence Platform (ECIP)

ECIP is a production-grade, highly scalable traffic-event management and decision-support system. Designed to assist urban traffic authorities, it forecasts the impact of planned and unplanned road incidents in Bengaluru, ranks response priorities, retrieves similar historical incidents for context, explains model outputs using SHAP values, and recommends optimal manpower and barricade allocations under constraints.

---

## 📑 Table of Contents
1. [Platform Architecture](#-platform-architecture)
2. [Core Modules](#-core-modules)
3. [Tech Stack](#-tech-stack)
4. [Dataset & Feature Engineering](#-dataset--feature-engineering)
5. [Project Structure](#-project-structure)
6. [Local Installation & Setup](#-local-installation--setup)
7. [Running the Application](#-running-the-application)
8. [API Documentation](#-api-documentation)
9. [Git & GitHub Configuration](#-git--github-configuration)

---

## 🏗 Platform Architecture

ECIP utilizes a modern, decoupled client-server architecture:

```
┌─────────────────────────────────────────────────────────────┐
│               WEB FRONTEND (Vanilla HTML5/CSS3/JS)          │
│  - Live Map View (Leaflet.js)                               │
│  - Interactive Prediction Form                              │
│  - Multi-Event Resource Allocation Interface                │
│  - Interactive What-If Scenario Sandbox                     │
│  - Dynamic SHAP Explanation Charts                          │
└──────────────┬──────────────────────────────▲───────────────┘
               │ HTTP POST / GET              │ JSON Responses
┌──────────────▼──────────────────────────────┴───────────────┐
│                 FASTAPI PYTHON BACKEND SERVICE              │
│                                                             │
│   ┌─────────────────────┐       ┌────────────────────────┐  │
│   │ Similar Event k-NN  │       │ CatBoost Models        │  │
│   │ (Historical Context)│       │ (Duration & Closure)   │  │
│   └──────────┬──────────┘       └───────────┬────────────┘  │
│              │                              │               │
│   ┌──────────▼──────────┐       ┌───────────▼────────────┐  │
│   │ Response Priority   │       │ Event Impact Index     │  │
│   │ Engine (P1-P4)      │       │ (EII Composite Score)  │  │
│   └──────────┬──────────┘       └───────────┬────────────┘  │
│              │                              │               │
│   ┌──────────▼──────────┐       ┌───────────▼────────────┐  │
│   │ OR-Tools Optimizer  │       │ SHAP Explainer         │  │
│   │ (ILP Resource Map)  │       │ (Feature Attribution)  │  │
│   └─────────────────────┘       └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Core Modules

### 1. Machine Learning Core (CatBoost)
- **Duration Model**: Regression CatBoost model predicting the exact event resolution time in minutes (using log-transformed duration target to handle right-skewed data).
- **Closure Model**: Binary classifier CatBoost model outputting the probability that an event will require a full road closure.

### 2. Event Impact Index (EII)
An objective, derived composite score ranging from `0` to `100` that quantifies incident severity without using synthetic targets:
$$\text{EII} = 0.40 \times \text{Duration Risk} + 0.35 \times \text{Closure Risk} + 0.10 \times \text{Priority Risk} + 0.15 \times \text{Location Risk}$$
- Output tiers: **Low** ($<25$), **Medium** ($25\text{--}49$), **High** ($50\text{--}74$), and **Critical** ($\ge 75$).

### 3. Response Priority Engine
An operations-driven prioritization classifier mapping incidents to response tiers:
- **Priority 1 (P1)**: Deploy immediately (e.g., high-risk unplanned events or Critical EII).
- **Priority 2 (P2)**: Deploy within 15 minutes.
- **Priority 3 (P3)**: Monitor.
- **Priority 4 (P4)**: Observe only.

### 4. Similar Event Intelligence
Weighted k-NN retrieval engine matching incoming incidents with historical records. It uses a **corridor + geohash** spatial filter to handle cold-start conditions and outputs average duration, road closure rate, and common resources deployed for similar events.

### 5. Multi-Event Resource Optimizer
An Integer Linear Programming (ILP) optimization solver built with **Google OR-Tools (SCIP)**. It allocates limited manpower (officers) and barricades across multiple active incidents to maximize priority coverage.

### 6. Explainable AI (SHAP Explanations)
Uses **SHAP (SHapley Additive exPlanations)** to output feature-importance bars, providing transparent explanations for why the model predicted specific durations or closure probabilities.

### 7. Scenario Planning Engine
A sandbox simulator that lets planners test "What-if?" questions (e.g., proactive road closures, scaling response units) and projects the revised EII impact based on calibrated elasticity values.

---

## 💻 Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Machine Learning**: CatBoost, SHAP, Scikit-learn, Pandas, NumPy
- **Optimization**: Google OR-Tools (SCIP solver)
- **Frontend**: HTML5, Vanilla CSS3 (Custom Premium Dark Theme), JavaScript (ES6+), Leaflet.js

---

## 📊 Dataset & Feature Engineering

ECIP trains and operates on a dataset containing **8,173 real-world traffic event records** from Bengaluru.

### Key Features Engineered:
- **Temporal**: Hour of day (sine/cosine transformation), day of week, weekend indicator, Indian holiday lookup.
- **Spatial**: Latitude, longitude, geohash level-7, corridor assignment.
- **Historical Metrics**: Corridor-wise average duration, closure rate, event density, and cause-wise aggregate duration.

---

## 📂 Project Structure

```
d:/Game/Flipkart/
├── Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv # Raw Dataset
├── requirements.txt         # Core dependencies
├── README.md                # Platform documentation (this file)
├── .gitignore               # Ignored local environments and temporary caches
│
└── ecip/
    ├── __init__.py
    ├── config.py            # Central config: weights, resource mapping, threshold values
    ├── saved_models/        # Folder holding serialized CatBoost checkpoints (.cbm)
    │   ├── duration_model.cbm
    │   └── closure_model.cbm
    │
    ├── data/                # Data pipelines
    │   ├── __init__.py
    │   ├── loader.py        # Loading, cleaning and parsing duration data
    │   └── features.py      # Feature engineering and geohashing
    │
    ├── models/              # ML components
    │   ├── __init__.py
    │   ├── duration_model.py
    │   ├── closure_model.py
    │   └── trainer.py       # Pipelines for model training and cross-validation
    │
    ├── core/                # System intelligence engines
    │   ├── __init__.py
    │   ├── eii.py           # Event Impact Index calculation
    │   ├── priority.py      # P1-P4 Priority assignment
    │   ├── similar_events.py # k-NN retrieval & aggregate statistics
    │   ├── scenario_planner.py # Elasticity simulator
    │   └── resource_optimizer.py # ILP OR-Tools optimizer
    │
    ├── api/                 # REST API endpoints
    │   ├── __init__.py
    │   ├── main.py          # FastAPI application bootstrapper
    │   ├── state.py         # Application state management (shared across endpoints)
    │   └── routes.py        # Routing tables (/predict, /explain, /similar, etc.)
    │
    └── dashboard/           # User interface
        ├── index.html       # Sidebar, 4 panel views, Leaflet layout
        ├── styles.css       # Standard dark-mode layout styling (no emojis, premium slate theme)
        └── app.js           # API integration, Leaflet controller, custom charts
```

---

## ⚙️ Local Installation & Setup

### 1. Prerequisites
- **Python 3.10+** (using virtual environment)
- **Windows OS**

### 2. Set Up Virtual Environment & Install Dependencies
Run the following commands in your PowerShell:

```powershell
# Navigate to the project directory
cd D:\Game\Flipkart

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install core dependencies
.\.venv\Scripts\pip.exe install -r ecip/requirements.txt
```

---

## 🚀 Running the Application

### Step 1: Train Models
Run the training pipeline. This will load the raw dataset, execute feature engineering, train CatBoost duration and closure models, validate them, and serialize models into `ecip/saved_models/`.

```powershell
.\.venv\Scripts\python.exe -m ecip.models.trainer
```

### Step 2: Launch the API Server
Start the Uvicorn-served FastAPI backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn ecip.api.main:app --reload --port 8000
```

### Step 3: Open the Dashboard
Navigate to [http://localhost:8000](http://localhost:8000) in your web browser. The frontend dashboard will load and connect to the active API.

---

## 🔌 API Documentation

Once the server is running, the interactive API documentation is accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Major Endpoints:
- `POST /api/v1/events/predict`: Completes the single-event prediction bundle. Returns predicted duration/closure, computed EII, response priority (P1-P4), 5 similar historical events, and pre-computed resource scenarios.
- `POST /api/v1/events/explain`: Generates SHAP explanation metrics for duration and closure.
- `POST /api/v1/optimize/resources`: Runs the OR-Tools solver to distribute officers/barricades across multiple active incidents.
- `POST /api/v1/events/scenario`: Runs scenario simulations for ad-hoc What-If queries.

---

## 🐙 Git & GitHub Configuration

To push this project to your GitHub repository, execute the following commands in PowerShell:

```powershell
# Initialize git repository
git init

# Add all files (excluding files listed in .gitignore like .venv/ and cache folders)
git add .

# Commit changes
git commit -m "Initial commit: Event-Driven Congestion Intelligence Platform (ECIP)"

# Link remote repository and push (replace URL with your target repo link)
git remote add origin https://github.com/YOUR_USERNAME/ECIP.git
git branch -M main
git push -u origin main
```
