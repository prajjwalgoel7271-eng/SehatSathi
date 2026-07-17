# SehatSaathi — AI-Powered Multi-Disease Screening Web App

SehatSaathi is an early-stage screening web application designed for non-invasive estimate risk indicators of Parkinson's, Anemia, and Tuberculosis.

## 🚀 Local Deployment

1. **Install Python 3.10+**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the local Flask server:**
   ```bash
   python app.py
   ```
4. **Access the application:**
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## ☁️ Deploying to Render

This application is configured for deployment on Render.

1. **Create a Web Service on Render:**
   - Link your Git repository containing these files.
2. **Configure Settings:**
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
3. **Add Environment Variables:**
   - `FLASK_SECRET_KEY`: (A random string value for session security)

### ⚠️ Production Storage Notice
The web application reads the baseline spiral training logs inside `spiral_dataset/` successfully. However, because Render's free tier environment has an **ephemeral filesystem**, any new drawing coordinates submitted during tests will not be permanently stored on disk across container restarts. To persist these across restarts, consider binding a persistent volume or an external SQL database.
