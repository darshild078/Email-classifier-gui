# 📖 Usage Guide

## GUI Framework Details

- **tkinter + ttk**: Native Python GUI toolkit for cross-platform interfaces
- **threading**: Perform model loading and analysis in background threads to keep UI responsive
- **pathlib & os**: Manage file paths and resource checks
- **logging**: Track UI events and errors for troubleshooting

## Application Workflow

1. **Startup**
   - Ensure `models/` folder contains `spam_classifier.pkl` and `feature_engineer.pkl` OR sample data at `data/samples/sample.csv`
   - Run:
     ```bash
     python run_app.py
     ```
   - Application window launches, model loading begins in background

2. **Main Screen**
   - **Input Panel**: Multi-line text area for message entry
   - **Demo Buttons**: “Ham Example” and “Spam Example” load random samples
   - **Analyze Message**: Button triggers spam detection
   - **Clear**: Clears text area and resets state

3. **Background Model Loading**
   - Runs in a separate thread
   - Status indicator shows “Loading...” then “Ready” or error
   - Analyze button disabled until model is ready

4. **Prediction**
   - On click, text is passed to `SpamPredictionEngine`
   - Engine preprocesses text, extracts features, and predicts
   - Results panel displays:
     - **Prediction**: HAM or SPAM (color-coded)
     - **Confidence**: Percentage score
     - **Detailed info**: Spam/Ham probabilities

## Sample Data Management

- **File**: `data/samples/sample.csv` with 10 ham and 10 spam samples
- **Loading**: pandas reads CSV into DataFrame
- **Filtering**: Separate ham and spam DataFrames
- **Random Sampling**:
  ```python
  df[df['label']==0].sample(1)
  df[df['label']==1].sample(1)
  ```
- **Demo Buttons** use pandas sampling to load messages without retraining

## Real-Time Analysis

- **Key Release Binding**: Optionally enable real-time analysis
- **Debouncing**: Analyze only after user stops typing (e.g., 1s delay)
- **Threading**: Each analysis run in background to prevent UI freeze

## Advanced Usage

### Batch Testing
- **Load CSV** of messages via file dialog
- **Process Batch**: Loop through messages, predict each, update progress bar
- **Export Results**: Save batch predictions to CSV

### Performance Metrics Display
- Show analysis history in table or chart
- Plot confidence over time using matplotlib embedded in GUI

## Integration Options

- **CLI**:
  ```bash
  python -c "from src.core.predictor import SpamPredictionEngine; print(SpamPredictionEngine().predict('Hello'))"
  ```
- **Web API**: Create REST endpoint using Flask or FastAPI wrapping `SpamPredictionEngine`
- **Database Logging**: Store predictions in SQLite or other DB for audit

## Customization Examples

- **Themes**: Modify `styles.py` color and font variables
- **Placeholder Text**: Change default prompt in `components.py`
- **Window Size**: Adjust geometry in `main_window.py`
- **Additional Buttons**: Add new sample categories (e.g., phishing)

---

**With this guide, end users can understand and use the GUI effectively, test sample data, run batch analyses, and integrate the spam detector into other workflows.**