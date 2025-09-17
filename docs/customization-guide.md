# 🔧 Customization Guide

This guide covers advanced customization of the Email Spam Detector using the implemented technologies.

## 1. Advanced Text Preprocessing

- **Custom Regex Patterns:** Extend `src/core/preprocessor.py`:
  ```python
  custom_patterns = {
      r'\bcrypto(?:currency)?\b': 'cryptocurrency',
      r'\b(u|ur)\b': 'you',
      r'\b\d{10,}\b': ' hasphone ',
  }
  preprocessor.add_custom_patterns(custom_patterns)
  ```
- **Multi-language Support:** Use NLTK stopwords per language:
  ```python
  from nltk.corpus import stopwords
  preprocessor.set_language_specific_rules(language='spanish')
  ```

## 2. Enhanced Feature Engineering

- **Linguistic Features:** Word count, sentence count, type-token ratio:
  ```python
  feat_engineer.use_linguistic_features = True
  ```
- **Behavioral Features:** Urgency, scarcity, social proof:
  ```python
  feat_engineer.use_behavioral_features = True
  ```

## 3. Ensemble Methods

- **Voting Classifier:** Soft voting of multiple models:
  ```python
  from core.ensemble_trainer import EnsembleSpamTrainer
  trainer = EnsembleSpamTrainer(ensemble_type='voting')
  ensemble = trainer.train_ensemble(X_train, y_train)
  ```
- **Stacking Classifier:** Meta-learner over base models:
  ```python
  trainer = EnsembleSpamTrainer(ensemble_type='stacking')
  ensemble = trainer.train_ensemble(X_train, y_train)
  ```

## 4. Neural Network Integration

- **Custom Neural Classifier:** Simple dense network in `src/core/neural_classifier.py`:
  ```python
  from core.neural_classifier import SimpleNeuralSpamClassifier
  nn = SimpleNeuralSpamClassifier(hidden_size=128)
  nn.fit(X_train, y_train)
  ```
- Supports `predict` and `predict_proba` for integration

## 5. Advanced GUI Components

- **Batch Processing Tab:** Load CSV, process and display results in table
- **Analytics Dashboard:** Real-time plotting with **matplotlib**:
  ```python
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(history_times, confidences)
  ```
- **Model Comparison:** Compare predictions of multiple models side by side

## 6. Real-Time Analysis & Model Comparison

- **Debounced Input:** Analyze after a delay when typing stops
- **Background Threads:** Use **threading** to run analysis without freezing GUI
- **Comparison Table:** Display results from Naive Bayes, ensemble, neural net

## 7. Technology Integration

- **matplotlib:** Embed plots via `FigureCanvasTkAgg`
- **threading:** Asynchronous model loading and analysis
- **tkinter.ttk:** Styled widgets and modern theme support

---

Customize these components in your codebase under `src/core/` and `src/gui/` to tailor the spam detector to your needs.