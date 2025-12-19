# 🚀 Deployment Guide

This guide explains how to deploy **CricOptima** to the web so you can showcase it in your portfolio.

## 🌟 Option 1: Streamlit Community Cloud (Recommended)

This is the easiest and fastest way to deploy your dashboard. It's free and connects directly to your GitHub repository.

### Prerequisites
1. Ensure your code is pushed to GitHub (which you have already done!).
2. Ensure `requirements.txt` is in the root folder (it is!).

### Steps to Deploy
1. **Sign Up/Login**: Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. **New App**: Click the **"New app"** button (top right).
3. **Connect Repository**:
    *   **Repository**: Select `millenniumsingha/CricOptima`.
    *   **Branch**: Select `master`.
    *   **Main file path**: Enter `app/streamlit_app.py`.
4. **Deploy**: Click **"Deploy!"**.

🎉 **That's it!** Streamlit will install the dependencies from `requirements.txt` and launch your app. You will get a unique URL (e.g., `https://cricoptima-demo.streamlit.app`) that you can link to in your portfolio.

---

## 🐳 Option 2: Docker (Advanced)

If you want to deploy the full stack (API + Dashboard) to a platform like Render, Railway, or AWS.

1. **Build Image**:
   ```bash
   docker build -t cricoptima .
   ```

2. **Run Locally**:
   ```bash
   docker-compose up
   ```

3. **Deploy to Cloud (e.g., Render)**:
   *   Connect your GitHub repo to Render.
   *   Select **"Docker"** as the environment.
   *   It will automatically build using the `Dockerfile`.

---

## 🖼️ Embedding in Your Portfolio

Once deployed on Streamlit Cloud, you can embed it in your portfolio website using an `iframe`:

```html
<iframe
  src="https://your-app-url.streamlit.app/?embed=true"
  height="800"
  style="width:100%;border:none;"
></iframe>
```
