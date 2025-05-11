import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu từ file CSV
data_path = "inverter_failure_simulated_data.csv"
df = pd.read_csv(data_path)
df = df.round(3)

# 2. Tách đặc trưng và nhãn
X = df.drop("is_failed_next_7_days", axis=1)
y = df["is_failed_next_7_days"]

# 3. Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Tối ưu hóa Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced']
}
model_lr = LogisticRegression(random_state=42)
grid = GridSearchCV(model_lr, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
model_lr = grid.best_estimator_

# 6. Đánh giá Logistic Regression
y_pred_lr = model_lr.predict(X_test_scaled)
y_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression ===")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
print(f"AUC Score: {roc_auc_score(y_test, y_prob_lr):.3f}")

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 7. Huấn luyện và đánh giá Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print(f"AUC Score: {roc_auc_score(y_test, y_prob_rf):.3f}")

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 8. Vẽ ROC so sánh
plt.figure(figsize=(6, 4))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'Logistic (AUC = {roc_auc_lr:.3f})')
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 9. Dự đoán với ví dụ cụ thể
example = pd.DataFrame({
    'ambient_temp': [42],
    'output_current': [7.5],
    'output_voltage': [395],
    'operation_hours': [6200],
    'failure_count_last_30_days': [3]
})

example_scaled = scaler.transform(example)
example_prob = model_lr.predict_proba(example_scaled)[0][1]
example_pred = model_lr.predict(example_scaled)[0]

print("\n--- Ví dụ cụ thể ---")
print("Thông số Inverter:")
print(example.to_string(index=False))
print(f"\nXác suất xảy ra lỗi trong 7 ngày tới (Logistic): {example_prob:.3f}")
print("Dự đoán (Logistic):", "Sẽ xảy ra lỗi" if example_pred == 1 else "Không có lỗi")

# 10. Vẽ phân bố lỗi thực tế
plt.figure(figsize=(6, 4))
plt.bar(['No Failure', 'Failure'], [sum(y_test == 0), sum(y_test == 1)], color=['green', 'red'])
plt.xlabel('Inverter Condition')
plt.ylabel('Sample Count')
plt.title('Actual Failure Distribution in Test Set')
plt.tight_layout()
plt.show()

# 11. Bảng kết quả dự báo Logistic Regression
results_df = X_test.copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = y_pred_lr
results_df["Probability"] = np.round(y_prob_lr, 3)
results_df["Correct"] = results_df["Actual"] == results_df["Predicted"]

print("\n=== Bảng kết quả dự báo - Logistic Regression ===")
print(results_df.head(10).to_string(index=False))

