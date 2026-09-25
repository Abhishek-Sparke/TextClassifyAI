# Deployment & Advanced Hybrid Embedding Guide

This guide details two advanced capabilities of the TextClassify AI platform:
1. **Dense Semantic Embeddings (Hybrid Sparse-Dense Pipeline)**
2. **Containerized Deployment to Google Cloud Run / AWS ECS & Vercel Integration**

---

## 1. 🧠 Dense Semantic Embeddings (Hybrid Sparse-Dense Pipeline)

### The Short-Text Challenge
In our forensic error analysis, documents under 25 words exhibited an error rate of **23.36%** (compared to **3.49%** for documents over 120 words). This occurs because sparse TF-IDF vectors rely on exact n-gram matching:
- A short sentence like *"cosmonaut lunar landing"* contains no exact n-gram overlap with a training document that uses *"NASA astronaut moon exploration"*, despite meaning the exact same thing.

### The Hybrid Solution
In `src/dense_embeddings.py`, we implement a **Hybrid Sparse-Dense Representation**:
$$\mathbf{x}_{\text{hybrid}} = \left[ \alpha \cdot \mathbf{x}_{\text{tfidf}} \;\Big|\; (1 - \alpha) \cdot \mathbf{x}_{\text{dense}} \right]$$

- **Sparse Component ($\mathbf{x}_{\text{tfidf}} \in \mathbb{R}^{5000}$):** Captures exact domain keywords, acronyms (`GPU`, `CAD`, `NASA`, `RBI`), and rare technical terms with high precision.
- **Dense Component ($\mathbf{x}_{\text{dense}} \in \mathbb{R}^{384}$):** Extracted via `sentence-transformers` (`all-MiniLM-L6-v2`), mapping sentences into a dense semantic metric space where cosine similarity captures conceptual intent.
- **Balancing Factor ($\alpha = 0.65$):** Retains keyword fidelity while providing a dense semantic safety net.

### Enabling Dense Embeddings
To enable dense embeddings in your environment:
```bash
# Install sentence-transformers (pulls PyTorch and huggingface-hub)
pip install sentence-transformers
```

Usage in Python:
```python
import joblib
from src.dense_embeddings import DenseEmbedder, HybridFeatureExtractor

# 1. Load fitted TF-IDF vectorizer
tfidf_vec = joblib.load("models/tfidf_vectorizer.joblib")

# 2. Initialize dense embedder
dense_embedder = DenseEmbedder(model_name="all-MiniLM-L6-v2")

# 3. Initialize hybrid extractor
hybrid_extractor = HybridFeatureExtractor(tfidf_vec, dense_embedder, alpha=0.65)

# 4. Transform documents into unified sparse-dense matrix
X_hybrid = hybrid_extractor.transform(["NASA launched a spacecraft.", "baseball game"])
print("Hybrid feature matrix shape:", X_hybrid.shape)  # (2, 5384)
```

---

## 2. ☁️ Containerized Deployment: Google Cloud Run

Google Cloud Run is the recommended platform for serverless container deployment: it automatically scales down to zero when idle (cost-effective) and scales up to handle thousands of requests per second.

### Prerequisites
- Google Cloud CLI (`gcloud`) installed and authenticated:
  ```bash
  gcloud auth login
  gcloud config set project YOUR_GCP_PROJECT_ID
  ```

### Automated Deployment
Execute the included deployment script:
```bash
# Linux / macOS:
chmod +x deploy_cloud_run.sh
./deploy_cloud_run.sh

# Windows (PowerShell):
.\deploy_cloud_run.ps1 -ProjectId YOUR_GCP_PROJECT_ID -Region us-central1
```

### Manual Step-by-Step Command Walkthrough
```bash
# 1. Build and push container to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/textclassify-api:v2.0.0 -f Dockerfile.api .

# 2. Deploy to Cloud Run
gcloud run deploy textclassify-api \
    --image gcr.io/YOUR_GCP_PROJECT_ID/textclassify-api:v2.0.0 \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --port 8000
```

Cloud Run will output your live URL:
`https://textclassify-api-xxxx-uc.a.run.app`

Verify deployment:
- Health check: `https://textclassify-api-xxxx-uc.a.run.app/health`
- Swagger UI: `https://textclassify-api-xxxx-uc.a.run.app/docs`

---

## 3. 📦 Containerized Deployment: AWS ECS (Fargate)

For AWS environments, deploy the API container to an AWS ECS Fargate cluster.

### Prerequisites
- AWS CLI configured with active credentials:
  ```bash
  aws configure
  ```
- Docker daemon running locally.

### Automated Deployment
```bash
chmod +x deploy_ecs.sh
./deploy_ecs.sh
```

---

## 4. 🌐 Connecting the Vercel Frontend

The React/Vite frontend is pre-configured to communicate with the deployed API while retaining zero-latency browser-side client execution if the API is offline.

### Step 1: Deploy Frontend to Vercel
You can deploy the frontend directly via Vercel CLI or GitHub integration:
```bash
# Deploy with Vercel CLI
cd frontend
vercel
```

### Step 2: Configure Environment Variable in Vercel
1. Go to the [Vercel Dashboard](https://vercel.com/dashboard).
2. Select your `TextClassify` project $\rightarrow$ **Settings** $\rightarrow$ **Environment Variables**.
3. Add the following variable:
   - **Key:** `VITE_API_URL`
   - **Value:** `https://textclassify-api-xxxx-uc.a.run.app` (your live Cloud Run or ECS URL)
   - **Environments:** Production, Preview, Development.
4. Redeploy the project on Vercel to bake the variable into the build bundle.

### Step 3: Verification
Open your Vercel deployment URL (`https://your-project.vercel.app`):
- Click **Analyze Document**.
- In the prediction card, notice the badge `Model: Logistic Regression (Cloud API)` indicating live server-side inference.
- If the Cloud API ever experiences network downtime, the frontend seamlessly fails over to the embedded client-side inference engine without disrupting the user experience!
