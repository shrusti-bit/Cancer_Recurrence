# 🏥 Advanced AI-Powered Thyroid Cancer Recurrence Prediction System

A comprehensive Responsible AI solution for predicting thyroid cancer recurrence with real-time predictions, bias auditing, and model interpretability features.

## 🌟 Key Features

### 🤖 AI Prediction Engine
- **Real-time predictions** using optimized XGBoost model (99.4% AUC)
- **Interactive patient input form** with comprehensive clinical features
- **Confidence scoring** and detailed clinical rationale
- **Risk stratification** with actionable recommendations

### 📊 Advanced Model Analytics
- **Performance metrics dashboard** with multiple visualization types
- **Confusion matrix analysis** with sensitivity/specificity calculations
- **ROC curve visualization** with AUC scoring
- **Model comparison charts** across different algorithms

### ⚖️ Comprehensive Bias Audit
- **Gender-based performance analysis** with fairness metrics
- **Age-group bias detection** across different demographics
- **Interactive fairness charts** using Plotly
- **Actionable bias mitigation recommendations**

### 📈 Model Interpretability
- **SHAP value analysis** for feature importance
- **Feature impact visualization** with color-coded charts
- **Decision process explanation** with hierarchical factor analysis
- **Interactive model behavior exploration**

## 🚀 Performance Metrics

| Metric | Score | Status |
|--------|-------|--------|
| **AUC Score** | 99.4% | Excellent |
| **F1-Score** | 95.2% | High |
| **Accuracy** | 97.4% | Very Good |
| **Precision** | 100% | Perfect |

## 🛠️ Technical Stack

- **Machine Learning**: XGBoost, Scikit-learn, SHAP
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Bias Auditing**: Fairlearn

## 📋 Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib seaborn shap fairlearn joblib
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Cancer_Recurrence_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Navigate through the different tabs to explore features

## 📁 Project Structure

```
Cancer_Recurrence_Project/
├── app.py                          # Main Streamlit application
├── optimized_xgb_model.joblib      # Trained XGBoost model
├── fitted_scaler.joblib            # Data preprocessing scaler
├── Thyroid_Diff.csv                # Original dataset
├── gender_fairness_metrics.csv     # Gender bias analysis results
├── age_fairness_metrics.csv        # Age bias analysis results
├── Untitled.ipynb                  # Jupyter notebook with model training
├── 01_PredictionEngine.html        # Static HTML prediction interface
├── 02_BiasAudit.html               # Static HTML bias audit interface
└── README.md                       # This file
```

## 🔬 Model Architecture

### Data Preprocessing
- **Feature Engineering**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for numerical features
- **Train/Test Split**: 80/20 stratified split

### Model Training
- **Algorithm**: XGBoost Classifier
- **Hyperparameter Tuning**: Grid search with 5-fold CV
- **Optimization Target**: F1-Score (handles class imbalance)
- **Feature Selection**: 39 engineered features from 15 original

### Model Validation
- **Cross-validation**: 5-fold stratified
- **Performance Metrics**: AUC, F1-Score, Accuracy, Precision
- **Bias Auditing**: Fairlearn framework for demographic analysis

## 📊 Dataset Information

- **Total Samples**: 383 patients
- **Features**: 15 clinical features
- **Target Variable**: Recurrence (28.2% positive cases)
- **Class Distribution**: Imbalanced (71.8% No Recurrence, 28.2% Recurrence)

### Key Features
- **Demographics**: Age, Gender
- **Clinical History**: Smoking, Radiation therapy
- **Thyroid Function**: Euthyroid, Hypothyroidism, Hyperthyroidism
- **Pathology**: Micropapillary, Papillary, Follicular
- **Staging**: TNM classification, Cancer stage
- **Treatment Response**: Excellent, Indeterminate, Structural Incomplete

## ⚖️ Bias Analysis Results

### Gender Analysis
- **Accuracy Gap**: 4.2% (Female: 98.3% vs Male: 94.1%)
- **Recall Gap**: 3.4% (Female: 92.3% vs Male: 88.9%)
- **Risk Level**: Low - Monitor for future updates

### Age Analysis
- **Accuracy Gap**: 4.5% (Young: 95.5% vs Others: 100%)
- **Recall Gap**: 22.2% (Young: 77.8% vs Others: 100%)
- **Risk Level**: High - Immediate action required

## 🎯 Usage Examples

### Making a Prediction
1. Navigate to the "🔮 AI Prediction Engine" tab
2. Fill in patient information:
   - Post-surgery response status
   - Initial risk assessment
   - Patient age and demographics
   - Additional clinical features
3. Click "Generate AI Prediction"
4. Review the risk assessment and clinical recommendations

### Analyzing Model Performance
1. Go to "📊 Model Analytics" tab
2. Explore different performance metrics
3. View confusion matrix and ROC curves
4. Compare with other algorithms

### Conducting Bias Audit
1. Access "⚖️ Bias Audit Dashboard"
2. Review gender and age-based performance
3. Analyze fairness metrics and gaps
4. Review recommendations for bias mitigation

## 🔧 Customization

### Adding New Features
1. Update the input form in `app.py`
2. Modify the feature mapping in the prediction logic
3. Retrain the model with new features
4. Update the SHAP analysis

### Modifying Visualizations
- Edit Plotly charts in the respective tab sections
- Customize colors and styling in the chart configurations
- Add new chart types as needed

## 📈 Future Enhancements

- [ ] **Real-time model retraining** with new data
- [ ] **API endpoints** for integration with hospital systems
- [ ] **Mobile-responsive design** for tablet use
- [ ] **Advanced bias mitigation** techniques
- [ ] **Multi-language support** for international deployment
- [ ] **Integration with EHR systems**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- **Dataset**: Thyroid cancer recurrence dataset
- **Libraries**: Streamlit, XGBoost, SHAP, Fairlearn
- **Community**: Open source ML community
- **Healthcare**: Medical professionals and researchers

---

**⚠️ Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used as the sole basis for clinical decision-making. Always consult with qualified healthcare professionals for medical advice and treatment decisions.


