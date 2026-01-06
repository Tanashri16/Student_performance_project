# 🎓 Student Analytics: Hybrid ML Performance Predictor

A data-driven Python application that combines **Linear Regression** and **Random Forest Classification** to predict student scores and identify "At-Risk" individuals while providing automated improvement suggestions.

## 🚀 Key Technical Features
* **Dual-Model Approach**: 
    * **Regression**: Predicts the exact `final_score` using Linear Regression.
    * **Classification**: Categorizes students as 'Safe' or 'At Risk' using a Random Forest Classifier.
* **Integrated SQL Database**: Uses `sqlite3` to manage and store student records efficiently from raw CSV data.
* **Automated Suggestion Engine**: A custom logic-based function that analyzes student metrics (attendance, study hours, etc.) to provide personalized improvement feedback.
* **Data Visualization**: Implements Seaborn heatmaps for correlation analysis to understand feature impact.

## 🛠️ Tech Stack
* **Language**: Python
* **Machine Learning**: Scikit-Learn (LinearRegression, RandomForestClassifier)
* **Data Handling**: Pandas, NumPy, SQLite3
* **Visualization**: Matplotlib, Seaborn

## 📁 Repository Structure
* `main.py`: The full pipeline including SQL integration, ML training, and prediction logic.
* `data/students.csv`: The source dataset containing student performance metrics.
* `student_data.db`: Automatically generated SQLite database for persistent storage.

## 📊 How It Works
1.  **Data Ingestion**: Reads `students.csv` and mirrors the data into a Local SQL database.
2.  **Feature Engineering**: Calculates a `pass_fail` target based on a 40-mark threshold.
3.  **Analytics**: Generates a correlation matrix to visualize relationships between attendance and scores.
4.  **Prediction**: 
    * Input: Attendance, Study Hours, Previous Marks, Assignments.
    * Output: Predicted score and Risk Status.
5.  **Advisory**: Outputs specific advice (e.g., "Improve attendance") based on performance thresholds.

## ⚙️ Installation
```bash
pip install pandas scikit-learn matplotlib seaborn
python main.py
