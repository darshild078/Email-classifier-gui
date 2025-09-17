# 📊 Dataset Guide

## 1. SMS Spam Collection Setup

**Source:** UCI Machine Learning Repository  
- [Dataset Info and Description](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Direct Download (smsspamcollection.zip)](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip)

**Setup Instructions:**
1. Download and extract the zip file:
   ```bash
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
   unzip smsspamcollection.zip
   ```
2. Rename and move the dataset for project use:
   ```bash
   mv SMSSpamCollection data/raw/spam.csv
   ```
3. The file should have two columns (tab-separated):
   ```txt
   ham\tCan we meet tomorrow for lunch?
   spam\tWINNER! Claim your $1000 now!
   ...
   ```

## 2. Alternative Datasets

- **Enron Email Dataset:** ([Download & Docs](https://www.cs.cmu.edu/~enron/))
    - 500,000+ emails for advanced email spam detection
    - Typically used for NLP research on email
- **SpamAssassin:** ([Public Corpus](https://spamassassin.apache.org/old/publiccorpus/))
    - Real-world ham and spam email samples
- **Custom Datasets:**
    - Any CSV with columns: `label,message`
    - Labels: 'ham' or 'spam'
    - Suitable for email, SMS, chat, or social data

## 3. Data Quality Insights & Preprocessing Requirements

- **Check for:**
    - Consistent labels ('ham'/'spam', lowercase)
    - No empty messages
    - Messages in appropriate language (English)
    - File is UTF-8 encoded
- **Preprocessing Steps:**
    - Remove/replace special characters, handle encodings
    - Filter out duplicates and extremely short/long messages
    - Preserve spam signals (URLs, emails, phones)
    - Normalize whitespace and casing
- **Best Practices:**
    - Balance dataset if possible (no extreme skew)
    - Remove personally identifiable info (PII)

## 4. Technologies for Data Handling

- **pandas**: Fast, flexible data loading and manipulation
- **pathlib**: Modern filesystem paths, cross-platform
- **os**: File and directory operations
- **logging**: Dataset loading and error tracking

## 5. Troubleshooting Common Dataset Issues

- **Problem:** Download fails
    - **Solution:** Use alternate links (wget/curl). Check your connection and try again.

- **Problem:** File encoding errors
    - **Solution:** Read file with correct encoding:
        ```python
        pd.read_csv('spam.csv', encoding='utf-8', sep='\t')
        # Try 'latin-1' if issues persist
        ```

- **Problem:** Wrong columns or labels
    - **Solution:** Ensure columns are `label,message` or convert them using pandas:
        ```python
        df.columns = ['label', 'message']
        df['label'] = df['label'].str.lower().str.strip()
        df = df[df['label'].isin(['ham', 'spam'])]
        ```

- **Problem:** Empty or corrupted rows
    - **Solution:**
        ```python
        df = df.dropna(subset=['message'])
        df = df[df['message'].str.strip() != '']
        ```

## 6. Validation Checklist & Ethics

- [ ] File exists in `data/raw/spam.csv` or your chosen location
- [ ] Contains clearly named `label` and `message` columns
- [ ] All labels are 'ham' or 'spam' (case-insensitive)
- [ ] There are no empty messages
- [ ] No PII or sensitive info present
- [ ] Language usage is appropriate for your application
- [ ] Dataset is balanced enough for model training
- [ ] You have rights to use and share the dataset
- [ ] You respect data privacy and ethical considerations

**Ethics Note:**
Always respect data privacy laws and terms of dataset licenses. Do not distribute datasets that contain personal or sensitive data.

---

**With this guide, your dataset preparation and troubleshooting should be smooth, secure, and reproducible. All technologies (pandas, pathlib, os, logging) are utilized through your code for best practice in dataset management.**
