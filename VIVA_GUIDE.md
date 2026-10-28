# SMS Shield – Viva Guide

Numbers below come from a real run of this project (5,160 messages after cleaning; 4,128 train / 1,032 test). Your app shows the live values on the **Model Performance** tab, so quote those if they differ slightly.

| Metric (SPAM class) | Result |
|---|---|
| Accuracy | 98.35% |
| Precision | 97.44% |
| Recall | 89.06% |
| F1-score | 93.06% |

Confusion matrix (test set): 901 legitimate correctly kept, 3 legitimate wrongly flagged as spam, 14 spam missed, 114 spam caught.

---

## 1. What is SMS spam detection?
Automatically deciding whether a text message is **spam** (unwanted ads, prize scams, phishing) or **legitimate** ("ham"). Example: "WINNER!! Claim your £900 prize, call now" → spam. "Are we meeting at 5 pm?" → legitimate.

## 2. Why did we choose this project?
Spam SMS causes annoyance and real financial fraud. It is a clear, real-world text-classification problem, the dataset is public and well known, and the model is simple enough to explain fully.

## 3. What is supervised machine learning?
The model learns from **labelled examples** (input + correct answer). Here every SMS is labelled `ham` or `spam`. The model finds patterns, then predicts labels for new messages.

## 4. What is text classification?
Assigning a category to a piece of text. Ours is **binary** classification (two classes: spam / ham).

## 5. What is TF-IDF?
A way to turn text into numbers. **TF** (term frequency) = how often a word appears in a message. **IDF** (inverse document frequency) = lowers the weight of words that appear in almost all messages. Words like "the" get a low score; words like "prize" or "claim" get a high score. In our code: `TfidfVectorizer(stop_words="english", ngram_range=(1, 2))`, which also uses two-word phrases such as "call now".

## 6. What is Multinomial Naive Bayes?
A probability classifier for word-count-like data. It learns how likely each word is in spam and in ham. For a new message it multiplies those probabilities (using Bayes' theorem) and picks the more probable class. "Naive" = it assumes words are independent. We use `alpha=0.1` (smoothing) so unseen words never give a zero probability.

## 7. Why is Naive Bayes suitable here?
- Fast to train and predict.
- Works well with many features (thousands of words) and moderate data.
- Handles text well even though the independence assumption is not perfectly true.
- Easy to explain, and we can show which words pushed a decision.

## 8. What are training data and testing data?
Training data (80%, 4,128 messages) is used to learn. Testing data (20%, 1,032 messages) is hidden during training and used only to measure performance on unseen messages, like an exam after studying.

## 9. Why 80:20?
It is a common balance: enough data to learn from, enough to test reliably. Too little training data → weak model; too little test data → unreliable score.

## 10. What is stratified sampling?
The split keeps the same spam/ham proportion in both sets (`stratify=labels`). Spam is only about 12% of our data, so without stratification the test set could by chance contain too few spam messages. `random_state=42` makes the split repeatable.

## 11. What is a scikit-learn Pipeline?
A chain of steps run in order as one object: `TfidfVectorizer → MultinomialNB`. Benefits: one `fit` and one `predict`, the same preprocessing is applied to new messages, and the vectorizer learns only from training data, which prevents **data leakage**.

## 12. Accuracy, precision, recall, F1
- **Accuracy**: correct predictions ÷ all predictions (98.35%). Can mislead because ham dominates.
- **Precision**: of messages called spam, how many truly were (97.44%). High precision = few good messages blocked.
- **Recall**: of all real spam, how many we caught (89.06%). High recall = little spam gets through.
- **F1-score**: balance of precision and recall (93.06%).

## 13. What is a confusion matrix?
A 2×2 table of actual vs predicted classes: true negatives (901), false positives (3), false negatives (14), true positives (114). It shows exactly which kinds of mistakes the model makes. Here it is cautious: it rarely blocks a genuine message, but misses some spam.

## 14. What is overfitting?
When a model memorises training data and performs poorly on new data. We guard against it by testing on a separate unseen test set, removing duplicate messages (so the same SMS is not in both sets), and using a simple model. Training and test performance being close indicates it is not badly overfitted.

## 15. What are the limitations?
- Relatively small, mostly English dataset.
- Bag-of-words ignores word order and deeper meaning.
- New spam tricks with unseen words can be missed (recall is 89%, not 100%).
- Cannot understand context, sarcasm or other languages.
- Predictions are probabilities, not certainty.

## 16. How does a new SMS get classified?
1. User types the message and clicks **Analyze Message**.
2. The saved pipeline converts it to a TF-IDF vector using the vocabulary learned in training.
3. Naive Bayes computes the spam and ham probabilities.
4. The higher one decides the label, shown as SPAM or LEGITIMATE with the confidence and the most influential words.

The model is **not retrained** on each click; it is cached (`st.cache_resource`).

## 17. How does Streamlit work?
Streamlit turns a Python script into a web app. The script runs from top to bottom each time the user interacts (button click, typing). Widgets like `st.text_area` and `st.button` return values, so no HTML/JavaScript server code is needed. Caching (`st.cache_data`, `st.cache_resource`) stops the data download and training from repeating. We added custom CSS for the design.

## 18. How is the application deployed?
1. Code is pushed to a public GitHub repository (`app.py`, `requirements.txt`, `README.md`).
2. On Streamlit Community Cloud we choose the repository, branch `main`, and main file `app.py`.
3. Streamlit installs the libraries from `requirements.txt` and runs the app.
4. We get a public URL. The dataset is downloaded automatically at start-up, so no local files, database or API keys are needed.

---

## Bonus questions

- **Why `stop_words="english"`?** Removes very common words that carry little meaning.
- **Why remove duplicates?** Identical messages in both train and test sets would inflate the score.
- **Why is accuracy not enough?** Always predicting "ham" would already give about 88% here, yet catch no spam.
- **Why spam as the positive class?** Our goal is to detect spam, so precision/recall are measured for it.
- **Which dataset?** UCI SMS Spam Collection (Almeida & Hidalgo), real SMS messages labelled ham/spam.
- **How could you improve it?** Try Logistic Regression or SVM, tune parameters, use more data, add deep-learning models.

---

## Presentation script (about 3 minutes)

"Good morning, respected faculty. My project is **SMS Shield, an SMS spam detection system using machine learning**.

**Problem.** Spam text messages are annoying and are often used for fraud, such as fake prize and phishing messages. Writing manual rules is difficult because spammers keep changing their wording, so I used machine learning.

**Dataset.** I used the UCI SMS Spam Collection, which contains real messages labelled spam or legitimate. After cleaning and removing duplicates, I used 5,160 messages, of which about 12% are spam.

**Method.** The data is split 80:20 into training and testing sets with stratification. Messages are converted into numbers using TF-IDF, which gives higher weight to meaningful words. Then a Multinomial Naive Bayes classifier learns which words are typical of spam. Both steps are combined in a scikit-learn pipeline.

**Results.** On messages the model never saw, it achieved about 98% accuracy, 97% precision, 89% recall and 93% F1-score. The confusion matrix shows it rarely blocks genuine messages, though it misses some spam.

**Application.** I built the web app using Streamlit. The user enters an SMS and the trained model predicts SPAM or LEGITIMATE, with probability and the words that influenced the decision. The app also shows performance metrics, dataset details and how the system works.

**Deployment.** The code is on GitHub and deployed free on Streamlit Community Cloud, so it can be used through a public link.

**Limitations and future work.** The dataset is small and mostly English, and the model does not understand context. In future I can compare other algorithms, use larger multilingual datasets and add user feedback.

Thank you. I will now demonstrate the application."
