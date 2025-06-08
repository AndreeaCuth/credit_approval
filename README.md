# Credit Approval ML Project

This project implements a machine learning pipeline to predict credit approval based on applicant data. 
It includes data preprocessing, model training, API deployment with FastAPI, and Docker containerization.

## Project Structure

```
credit_approval/
├── data/
│   └── credit_dataset.csv         # Raw dataset
├── experiments/
│   └── credit_approval_exp.ipynb  # Notebook with EDA and training
├── model/
│   ├── credit_approval_model.pkl  # Trained ML model
│   └── label_encoders.pkl         # Encoders used for categorical features
├── src/
│   └── credit_approval.py         # Module with preprocessing or helper logic
├── main.py                        # FastAPI app exposing the model as an API
├── Dockerfile                     # Docker container definition
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/AndreeaCuth/credit_approval.git
cd credit_approval
```

### 2. Create Virtual Environment (optional)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI App

```bash
uvicorn main:app --reload
```

Access the API documentation at:  
http://127.0.0.1:8000/docs

## Docker

### Build the image:

```bash
docker build -t credit_approval_app .
```

### Run the container:

```bash
docker run -p 8000:8000 credit_approval_app
```

## Model Details

- Models evaluated: Logistic Regression, Random Forest, XGBoost, LightGBM, KNN, SVM
- Final model: `RandomForestClassifier` (best performance)
- Input features were preprocessed and encoded accordingly
- Outputs: binary classification – Approved (1) or Not Approved (0)

## Input Format (API)

```json
{
    Gender: int              
    Age: float
    Debt: float
    Married: int           
    BankCustomer: int       
    Industry: str            
    Ethnicity: str           
    YearsEmployed: float
    PriorDefault: int       
    Employed: int           
    CreditScore: int
    DriversLicense: int      
    Citizen: str             
    ZipCode: int
    Income: float
}
```

## Output Format

```json
{
  "prediction": string Approved/Rejected,
  "probability": float <score>,
}
```


