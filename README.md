# 💧 HYDROWATCH-A WATER QUALITY MONITORING NETWORK FOR DRINKING AND IRRIGATION 

An interactive machine learning-based application for predicting water quality using various chemical parameters. This app supports both **classification** (predicting water quality class) and **regression** (predicting Water Quality Index - WQI) tasks using models like Random Forest, XGBoost, CatBoost, and more — all served through a **Gradio** interface.

---

## 🚀 Features

- 🔄 **Switch between classification and regression**
- 🧠 **Trains multiple ML models automatically**
  - Random Forest, XGBoost, CatBoost, SVM, Naive Bayes, Logistic Regression, etc.
- 📈 **Performance metrics and visual comparison**
- 🔮 **Real-time prediction** from user input
- 📊 **Beautiful visualizations** (accuracy, precision, R², MAE)
- 🧩 Easy to use Gradio UI

---

## 🧪 Dataset Info

The dataset should include the following water quality features:

- `pH`, `EC`, `CO3`, `HCO3`, `Cl`, `SO4`, `NO3`, `TH`, `Ca`, `Mg`, `Na`, `K`, `F`, `TDS`

Target columns:
- `Water Quality Classification` (for classification)
- `WQI` (for regression)

> 🗂 Place your `DataSet.csv` in the root directory or adjust the path in `app.py`.

---

## 🛠 Tech Stack

- **Frontend:** Gradio
- **Backend/ML:** Python, Scikit-learn, XGBoost, CatBoost
- **Visualization:** Seaborn, Matplotlib
- **Data Handling:** Pandas, Numpy

---

## 📂 Project Structure
-HYDROWATCH-A WATER QUALITY MONITORING NETWORK FOR DRINKING AND IRRIGATION /
-├── app.py # Main Gradio ML application
-├── DataSet.csv # Dataset file (used in training)
-├── requirements.txt # List of required Python packages
-├── README.md # Project documentation (you’re reading it)

🧪 How to Run
**1.Clone the repository:

git clone https://github.com/yourusername/water-quality-predictor.git
cd water-quality-predictor
**2.Install dependencies:

pip install -r requirements.txt
**3.Make sure the dataset is present at the specified path.
**4.Run the application:
python app.py
**5.The app will launch in your browser.


Copyright (c) 2025 Aditya Sripalle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     
copies of the Software, and to permit persons to whom the Software is         
furnished to do so, subject to the following conditions:                      

The above copyright notice and this permission notice shall be included in   
all copies or substantial portions of the Software.                           

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR   
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,     
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.

