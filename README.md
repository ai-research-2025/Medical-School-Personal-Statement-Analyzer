# Medical School Personal Statement Analyzer

An AI-powered tool for analyzing and scoring medical school personal statements based on established rubrics. The system evaluates statements across four key categories using state-of-the-art natural language processing models.

## Overview

This application uses machine learning to automatically analyze personal statements for medical school applications, providing detailed feedback and scoring based on established rubrics. The system segments text, classifies content, and provides scores along with explanatory feedback.

## Key Features

- **Automated Text Analysis**: Segments personal statements and classifies content into key categories
- **Rubric-Based Scoring**: Evaluates content on a 1-4 scale based on established medical school admission criteria  
- **Multiple Model Support**: Includes 5 state-of-the-art embedding models for comparison
- **Detailed Feedback**: Provides segment-by-segment analysis with improvement recommendations
- **Batch Processing**: Analyze multiple statements simultaneously
- **PDF Reports**: Generate professional analysis reports

## Categories Evaluated

The system evaluates personal statements across four key categories:

### 1. Spark (Opening)
- **Description**: Opening that spurs interest in medicine (typically in opening paragraph)
- **Score 4**: Engaging and logically flows into becoming a doctor
- **Score 1**: Disconnected from being a doctor or confusing/random

### 2. Healthcare Experience  
- **Description**: Watching/participating in healthcare - medical professional at work
- **Score 4**: Vivid, active, thoughtful, relevant, memorable, positive and optimistic
- **Score 1**: Passive observation, uninteresting, irrelevant, problematic, negative tone

### 3. Showing Doctor Qualities
- **Description**: Stories/examples portraying vision of doctor role and appealing aspects  
- **Score 4**: Realistic, self-aware, mature, humble, specific and clear understanding, positive
- **Score 1**: Arrogant, immature, overly confident, inaccurate understanding, negative tone

### 4. Spin (Connection)
- **Description**: Explaining why experiences qualify them to be a doctor
- **Score 4**: Direct, logical, and specific argument connecting experience to profession
- **Score 1**: Brief, vague, simplistic connection to being a doctor, generic

## Supported Models

The application includes five embedding models for text analysis:

1. **BAAI/bge-large-en-v1.5** (Default - State-of-the-art)
2. **intfloat/e5-large-v2** (Excellent alternative)
3. **thenlper/gte-large** (Top performer)
4. **all-mpnet-base-v2** (Good fallback)
5. **all-MiniLM-L6-v2** (Lightweight fallback)

## Installation

### Prerequisites
- Python 3.8+
- Required packages (install via pip):

```bash
pip install streamlit pandas numpy scikit-learn torch sentence-transformers xgboost reportlab openpyxl
```

### Setup
1. Clone or download the repository
2. Install dependencies
3. Place training data files in the main directory:
   - `DedooseChartExcerpts_2025_8_5_1025.xlsx`
   - `Personal Statements Coded.xlsx`

## Usage

### Step 1: Train the AI Models

1. **Launch the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Navigate to "Step 1: Train Model"** tab

3. **Upload Training Files**:
   - Upload both Excel files containing manually coded personal statements
   - The system expects files with coded excerpts and rubric scores

4. **Select Model** (Optional):
   - The code defaults to `bge-large-en-v1.5` (state-of-the-art)
   - To test other models, modify the model selection in the code

5. **Click "Start Training"**:
   - The system will extract features from training data
   - Train ensemble classifiers for each category
   - Save trained models to `trained_models/` directory
   - Display training accuracy for each category

6. **Record Results**:
   - Note the training accuracy displayed for each category
   - Compare results across different models

### Step 2: Analyze Personal Statements

1. **Navigate to "Step 2: Analyze Statements"** tab

2. **Select Input Method**:
   - **Single Statement**: Upload a `.txt` file with one complete statement
   - **Multiple Statements**: Upload an Excel/CSV file with multiple statements

3. **Run Analysis**:
   - Click analyze to process the statement(s)
   - View detailed segment-by-segment results
   - Review category scores and confidence levels

4. **Review Results**:
   - **Overall Summary**: Categories found, average scores, confidence levels
   - **Detailed Analysis**: Segment-by-segment breakdown with explanations
   - **Recommendations**: Specific suggestions for improvement
   - **Probability Distribution**: Model confidence across all categories

5. **Export Results**:
   - Download PDF reports for detailed analysis
   - Export JSON/CSV data for batch processing

### Step 3: View Rubrics

1. **Navigate to "Step 3: View Rubrics"** tab
2. **Review Scoring Criteria**: Detailed rubrics for each category (1-4 scale)
3. **Study Examples**: Keywords, patterns, and quality indicators
4. **Implementation Tips**: Guidance for achieving high scores

## Testing Different Models

To compare the performance of different embedding models:

1. **Modify Model Selection**: 
   - Edit `app.py` line 307-311 to change the model priority order
   - Or modify the `load_sentence_transformer()` function to use a specific model

2. **Train Each Model**:
   - Train with `bge-large-en-v1.5` (default)
   - Change code to use `intfloat/e5-large-v2` and retrain  
   - Repeat for `thenlper/gte-large`, `all-mpnet-base-v2`, and `all-MiniLM-L6-v2`

3. **Record Performance**:
   - Training accuracy for each category
   - Overall model accuracy
   - Analysis quality on test scripts

4. **Test on Sample Data**:
   - Use files in `test_scripts/` folder (22 coded personal statements)
   - Compare classification results and scores across models
   - Evaluate consistency and accuracy

## File Structure

```
personal_statements/
├── app.py                          # Main Streamlit application
├── trained_models/                 # Saved model files
│   ├── classifier.pkl             # Trained classifiers
│   ├── embedder_name.txt          # Active model name
│   ├── ensemble.pkl               # Ensemble model
│   ├── scaler.pkl                 # Feature scaler
│   ├── scorer.pkl                 # Scoring models
│   └── thresholds.pkl             # Classification thresholds
├── test_scripts/                   # Test personal statements (22 files)
│   ├── Anne Coded.txt
│   ├── Brandt Coded.txt
│   └── ...
├── DedooseChartExcerpts_2025_8_5_1025.xlsx    # Training data 1
├── Personal Statements Coded.xlsx              # Training data 2
├── Coding Scheme + Rubric--Personal Statements.pdf  # Rubric documentation
└── README.md                       # This file
```

## Technical Details

### Architecture
- **Text Processing**: Advanced semantic segmentation using sentence transformers
- **Feature Extraction**: Combines semantic embeddings with rubric-specific features
- **Classification**: Ensemble of Random Forest, XGBoost, and Neural Network models
- **Scoring**: Separate regression models for 1-4 scale scoring
- **Confidence**: Ensemble voting and probability analysis

### Model Training Process
1. **Data Loading**: Combines and preprocesses training data from Excel files
2. **Feature Engineering**: Extracts semantic and rubric-aligned features
3. **Model Training**: Trains separate classifiers and scorers for each category
4. **Ensemble Creation**: Combines models for improved performance
5. **Threshold Optimization**: Calibrates classification thresholds
6. **Model Persistence**: Saves all components for analysis

### Performance Metrics
- **Classification Accuracy**: Category detection accuracy
- **Scoring Accuracy**: 1-4 scale prediction accuracy  
- **Confidence Calibration**: Probability alignment with actual performance
- **Ensemble Performance**: Combined model effectiveness

## Authors

Developed by: Faith Marie Kurtyka, Cole Krudwig, Sean Dore, Sara Avila, George (Guy) McHendry, Steven Fernandes

## Support

For issues or questions:
1. Check the rubric documentation in `Coding Scheme + Rubric--Personal Statements.pdf`
2. Review training data format in the Excel files
3. Test with sample files in `test_scripts/` folder
4. Verify model training completed successfully in Step 1

## Model Comparison Results

When testing different models, record results in the following format:

| Model | Spark Accuracy | Healthcare Accuracy | Doctor Qualities Accuracy | Spin Accuracy | Overall Accuracy |
|-------|----------------|---------------------|---------------------------|---------------|------------------|
| bge-large-en-v1.5 | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% |
| intfloat/e5-large-v2 | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% |
| thenlper/gte-large | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% |
| all-mpnet-base-v2 | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% |
| all-MiniLM-L6-v2 | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% |

Use the test scripts to evaluate model performance and compare classification consistency across different embedding approaches.
