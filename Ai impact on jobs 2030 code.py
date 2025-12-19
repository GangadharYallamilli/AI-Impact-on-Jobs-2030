import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv(r"D:\My Data Sets\AI_Impact_on_Jobs_2030.csv")

target_regression = 'Automation_Probability_2030'
target_classification = 'Risk_Category'

numerical_features = [
    'Average_Salary', 'Years_Experience',
    'AI_Exposure_Index', 'Tech_Growth_Factor'
] + [f'Skill_{i}' for i in range(1, 11)]

categorical_features = ['Education_Level']

all_features = df.drop(
    columns=[target_regression, target_classification, 'Job_Title']
).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ---------------- ML MODELS ----------------
X = df[all_features]
y = df[target_regression]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_p, y_train)
y_pred = reg.predict(X_test_p)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)

# Classification
y_cls = df[target_classification]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_cls, test_size=0.3, random_state=42
)

X_train_cp = preprocessor.fit_transform(X_train_c)
X_test_cp = preprocessor.transform(X_test_c)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_cp, y_train_c)
y_pred_c = clf.predict(X_test_cp)

ACC = accuracy_score(y_test_c, y_pred_c)

# Clustering
skills = df[[f'Skill_{i}' for i in range(1, 11)]]
skills_scaled = StandardScaler().fit_transform(skills)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(skills_scaled)

# ---------------- PLOTS ----------------
def heatmap():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_features + [target_regression]].corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def distribution():
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_regression], kde=True)
    plt.title("Automation Probability Distribution")
    plt.show()

def salary_box():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Education_Level", y="Average_Salary", data=df)
    plt.title("Salary by Education Level")
    plt.show()

def scatter():
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="AI_Exposure_Index",
        y=target_regression,
        hue=target_classification,
        data=df
    )
    plt.title("AI Exposure vs Automation Risk")
    plt.show()

def clusters():
    cluster_avg = df.groupby("Cluster")[target_regression].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Cluster", y=target_regression, data=cluster_avg)
    plt.title("Automation Risk by Skill Cluster")
    plt.show()

# ---------------- GUI ----------------
root = tk.Tk()
root.title("AI Impact on Jobs – 2030")
root.geometry("1100x650")
root.configure(bg="#0f172a")

tk.Label(
    root,
    text="🤖 AI Impact on Jobs – 2030",
    bg="#020617",
    fg="#38bdf8",
    font=("Segoe UI", 22, "bold"),
    pady=15
).pack(fill="x")

main = tk.Frame(root, bg="#0f172a")
main.pack(expand=True, fill="both")

sidebar = tk.Frame(main, bg="#020617", width=250)
sidebar.pack(side="left", fill="y")

content = tk.Frame(main, bg="#0f172a")
content.pack(side="right", expand=True, fill="both")

def clear():
    for w in content.winfo_children():
        w.destroy()

# ---------------- PAGES ----------------
def dashboard():
    clear()
    box = tk.Frame(content, bg="#0f172a")
    box.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(box, text="Dashboard",
             bg="#0f172a", fg="#38bdf8",
             font=("Segoe UI", 20, "bold")).pack(pady=15)

    tk.Label(box, text=f"R² Score : {R2:.3f}",
             bg="#0f172a", fg="white",
             font=("Segoe UI", 14)).pack(pady=5)

    tk.Label(box, text=f"RMSE : {RMSE:.3f}",
             bg="#0f172a", fg="white",
             font=("Segoe UI", 14)).pack(pady=5)

    tk.Label(box, text=f"Accuracy : {ACC:.3f}",
             bg="#0f172a", fg="white",
             font=("Segoe UI", 14)).pack(pady=5)

def eda():
    clear()
    box = tk.Frame(content, bg="#0f172a")
    box.place(relx=0.5, rely=0.5, anchor="center")

    tk.Button(box, text="Correlation Heatmap", width=25, command=heatmap).pack(pady=8)
    tk.Button(box, text="Automation Distribution", width=25, command=distribution).pack(pady=8)
    tk.Button(box, text="Salary Box Plot", width=25, command=salary_box).pack(pady=8)
    tk.Button(box, text="AI Exposure Scatter", width=25, command=scatter).pack(pady=8)

def clustering_page():
    clear()
    box = tk.Frame(content, bg="#0f172a")
    box.place(relx=0.5, rely=0.5, anchor="center")

    tk.Button(box, text="Show Skill Clusters", width=25, command=clusters).pack()

# ---------------- SIDEBAR ----------------
btn = {"bg": "#020617", "fg": "white", "font": ("Segoe UI", 12), "bd": 0}

tk.Button(sidebar, text="🏠 Dashboard", command=dashboard, **btn).pack(fill="x", pady=10)
tk.Button(sidebar, text="📊 EDA", command=eda, **btn).pack(fill="x", pady=10)
tk.Button(sidebar, text="🔍 Clustering", command=clustering_page, **btn).pack(fill="x", pady=10)

dashboard()
root.mainloop()
