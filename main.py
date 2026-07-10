import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#  Read CSV
df = pd.read_csv("data/students.csv")
print("CSV Data:")
print(df)

# Connect to SQL & store data
conn = sqlite3.connect("student_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    attendance REAL,
    study_hours REAL,
    previous_marks REAL,
    assignments INTEGER,
    final_score REAL
)
""")
conn.commit()

# Insert CSV data into SQL
df.to_sql("students", conn, if_exists="replace", index=False)

# Fetch from SQL
df_sql = pd.read_sql("SELECT * FROM students", conn)
print("\nData from SQL:")
print(df_sql)

# Correlation Analysis (Optional)
corr = df_sql.corr()
print("\nCorrelation matrix:")
print(corr)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

#  Regression → Predict Marks
X = df_sql[['attendance', 'study_hours', 'previous_marks', 'assignments']]
y = df_sql['final_score']

reg = LinearRegression()
reg.fit(X, y)

new_student_df = pd.DataFrame([[80, 3.5, 70, 8]], columns=X.columns)
predicted_marks = reg.predict(new_student_df)
print("\nPredicted Marks:", predicted_marks[0])

print("\nFeature Importance (Regression Coefficients):")
for feature, coef in zip(X.columns, reg.coef_):
    print(feature, ":", coef)

# Classification → Pass/Fail
df_sql['pass_fail'] = df_sql['final_score'].apply(lambda x: 1 if x >= 40 else 0)

# Features & target
X_class = df_sql[['attendance', 'study_hours', 'previous_marks', 'assignments']]
y_class = df_sql['pass_fail']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Calculate Classification Accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nClassification Accuracy:", round(accuracy * 100, 2), "%")

# Predict Pass/Fail for a new student
new_student_df_class = pd.DataFrame([[60, 2.0, 55, 5]], columns=X_class.columns)
prediction = clf.predict(new_student_df_class)

if prediction[0] == 1:
    print("\nStudent is Safe")
else:
    print("\nStudent is At Risk")

# Feature importance
print("\nFeature Importance (Classification):")
for feature, importance in zip(X_class.columns, clf.feature_importances_):
    print(feature, ":", importance)

#  STEP 9: Suggest Improvement Areas 
def suggest_improvements(row):
    improvements = []
    if row['attendance'] < 75:
        improvements.append("Improve attendance")
    if row['study_hours'] < 3:
        improvements.append("Study more hours")
    if row['previous_marks'] < 60:
        improvements.append("Focus on weak subjects")
    if row['assignments'] < 6:
        improvements.append("Complete more assignments")
    
    if len(improvements) == 0:
        return "Keep up the good work"
    else:
        return ", ".join(improvements)

df_sql['improvement_suggestions'] = df_sql.apply(suggest_improvements, axis=1)

print("\nStudent Suggestions:")
print(df_sql[['attendance', 'study_hours', 'previous_marks', 'assignments', 'final_score', 'pass_fail', 'improvement_suggestions']])


