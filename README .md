# 🛡️ SMS Shield – SMS Spam Detection Using Machine Learning

SMS Shield is a web application that classifies an SMS message as **SPAM** or **LEGITIMATE** using a machine learning model trained on the UCI SMS Spam Collection dataset. It is built with Python, scikit-learn and Streamlit.

## Features

- Real predictions from a trained TF-IDF + Multinomial Naive Bayes pipeline
- Spam / legitimate probability and the words that influenced the decision
- Sample messages, input validation and a clear/try-another option
- Real test-set metrics: accuracy, precision, recall, F1-score
- Confusion matrix heatmap and dataset class-distribution chart
- "How it works" and "About" sections suitable for a project presentation
- Dataset downloads automatically; model is cached and trained only once
- No database, no API keys, no paid services

## Technologies Used

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit · GitHub · Streamlit Community Cloud

## Project Structure

```
sms-shield/
├── app.py            # Complete Streamlit app (data, model, metrics, UI)
├── requirements.txt  # Python libraries
├── README.md         # This file
└── VIVA_GUIDE.md     # Viva questions, answers and presentation script
```

## How the ML part works

1. The dataset is downloaded from the UCI repository (a GitHub mirror is used as backup).
2. Invalid rows, empty messages and duplicate messages are removed.
3. Data is split 80% train / 20% test (`random_state=42`, stratified).
4. A scikit-learn `Pipeline` does `TfidfVectorizer` → `MultinomialNB`, fitted on training data only.
5. Metrics are computed from the model's predictions on the unseen test set.
6. Every message typed by the user goes through the same trained pipeline.

## Installation (local)

1. Install **Python 3.10 or newer** from https://www.python.org/downloads/ (tick **"Add Python to PATH"** on Windows).
2. Open a terminal in the project folder and create a virtual environment:

   ```bash
   python -m venv venv
   ```

   Activate it:

   - Windows: `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`

3. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Run locally

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`. The first start takes a few seconds because the dataset is downloaded and the model is trained. After that it is cached.

## Test the model

1. Open the **Analyze SMS** tab and click the sample "🎁 Prize scam" button, then **Analyze Message**. It should be classified as SPAM.
2. Click "👋 Friendly chat" and analyze. It should be LEGITIMATE.
3. Try your own messages. Clicking Analyze with an empty box shows a warning.
4. Open **Model Performance** to see the metrics and confusion matrix computed on the test set.

## Upload to GitHub

### Option A – through the website (easiest)

1. Create an account at https://github.com and click **New repository**.
2. Name it `sms-shield`, choose **Public**, and click **Create repository**.
3. Click **uploading an existing file**, drag in `app.py`, `requirements.txt`, `README.md` and `VIVA_GUIDE.md` (the files themselves, not a folder inside a folder).
4. Click **Commit changes**.

### Option B – using Git

```bash
git init
git add app.py requirements.txt README.md VIVA_GUIDE.md
git commit -m "SMS Shield ML project"
git branch -M main
git remote add origin https://github.com/<your-username>/sms-shield.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account (authorize access if asked).
2. Click **Create app** (or **New app**) and choose to deploy from an existing GitHub repository.
3. Select the repository `sms-shield`, branch `main`, and set **Main file path** to `app.py`.
4. (Optional) choose a custom app URL such as `sms-shield-yourname`.
5. Click **Deploy**. The first build takes a few minutes while libraries install.
6. When it finishes you get a public link like `https://sms-shield-yourname.streamlit.app`. Share this link.

Any time you push a change to GitHub, the app updates automatically.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognized | Activate the virtual environment, or run `python -m streamlit run app.py` |
| `ModuleNotFoundError` locally or on the cloud | Make sure the library is listed in `requirements.txt`, then run `pip install -r requirements.txt` or redeploy |
| Streamlit Cloud says it cannot find `app.py` | Check the main file path is exactly `app.py` and the file is in the repository root, not inside a subfolder |
| "Could not load the dataset" message | The app needs internet to download the data. Check your connection and press **Retry** |
| Dependency install fails on the cloud | In app **Settings → Advanced**, choose Python 3.11 or 3.12 and reboot the app |
| App shows old content after an update | Open the app menu → **Reboot app**, or press `C` then clear cache |
| Repository is private and not visible | Give Streamlit access to the repo in GitHub settings, or make the repo public |

## Disclaimer

Predictions come from a statistical model trained on a limited dataset. They are estimates and may occasionally be incorrect. This project is for educational purposes.

## Dataset

UCI Machine Learning Repository – SMS Spam Collection (Almeida & Hidalgo): https://archive.ics.uci.edu/dataset/228/sms+spam+collection
