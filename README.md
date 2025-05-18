# Laptop Price Predictor

A machine learning project that predicts laptop prices based on specifications using a supervised learning approach.

## Project Overview

This project implements a full-stack machine learning solution that predicts laptop prices based on various features including:
- Brand
- Type (Notebook, Gaming, etc.)
- RAM
- Operating System
- Weight
- Touchscreen capability
- IPS Display
- Screen Size and Resolution
- CPU specifications
- Storage (HDD/SSD)
- GPU specifications

## Project Structure

```
laptop_price_predictor/
├── data/                      # Data files
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
├── models/                   # Trained models
├── notebooks/               # Jupyter notebooks for EDA
├── src/                     # Source code
│   ├── data/               # Data processing scripts
│   ├── features/           # Feature engineering
│   ├── models/             # Model training scripts
│   └── visualization/      # Visualization scripts
├── streamlit_app/          # Streamlit web application
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd laptop_price_predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing and Model Training
1. Place the raw dataset in `data/raw/`
2. Run the preprocessing scripts:
```bash
python src/data/preprocess.py
```

3. Train the model:
```bash
python src/models/train_model.py
```

### Running the Web Application
```bash
cd streamlit_app
streamlit run app.py
```

## Features

1. **Data Processing**
   - Comprehensive data cleaning and preprocessing
   - Feature engineering
   - Handling categorical variables

2. **Exploratory Data Analysis**
   - Distribution analysis
   - Correlation studies
   - Feature importance visualization

3. **Machine Learning**
   - Model selection and training
   - Hyperparameter optimization
   - Performance evaluation

4. **Web Interface**
   - User-friendly input form
   - Real-time price predictions
   - Interactive visualizations

## Technologies Used

- Python 3.8+
- Pandas, NumPy for data processing
- Scikit-learn for machine learning
- Matplotlib, Seaborn for visualization
- Streamlit for web interface
- Plotly for interactive visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 