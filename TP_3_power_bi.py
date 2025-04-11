import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import numpy as np

# تحميل البيانات من Kaggle
os.system("kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store")
os.system("unzip ecommerce-behavior-data-from-multi-category-store.zip")

# تحديد الملف الرئيسي
file_path = "2019-Nov.csv"

# قراءة البيانات على شكل أجزاء
chunksize = 100000
cleaned_data = []
df_chunks = pd.read_csv(file_path, chunksize=chunksize)

for chunk in df_chunks:
    chunk['event_time'] = pd.to_datetime(chunk['event_time'], errors='coerce')
    chunk.dropna(inplace=True)
    chunk.drop_duplicates(inplace=True)
    chunk = chunk[chunk['price'] > 0]
    cleaned_data.append(chunk)

df = pd.concat(cleaned_data, ignore_index=True)
df.to_csv("cleaned_data.csv", index=False)
print("✅ تم تنظيف البيانات وحفظها في cleaned_data.csv")

# تحميل البيانات المنظفة
cleaned_data = pd.read_csv("cleaned_data.csv")

# تحضير بيانات العملاء
customer_activity = cleaned_data.groupby('user_id').agg(
    total_purchases=('event_type', lambda x: (x == 'purchase').sum()),
    total_carts=('event_type', lambda x: (x == 'cart').sum()),
    total_views=('event_type', lambda x: (x == 'view').sum()),
    total_spent=('price', 'sum')
).reset_index()

# تطبيق K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
customer_activity['cluster'] = kmeans.fit_predict(customer_activity[['total_purchases', 'total_carts', 'total_views', 'total_spent']])

# تصور التجزئة
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_spent', y='total_purchases', hue='cluster', data=customer_activity, palette='viridis')
plt.title('تجزئة العملاء بناءً على النشاط والإنفاق')
plt.xlabel('إجمالي الإنفاق')
plt.ylabel('عدد عمليات الشراء')
plt.show()

# تدريب نموذج تصنيف باستخدام RandomForest
customer_activity['has_purchased'] = customer_activity['total_purchases'].apply(lambda x: 1 if x > 0 else 0)
X = customer_activity[['total_carts', 'total_views', 'total_spent']]
y = customer_activity['has_purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"دقة النموذج: {accuracy_score(y_test, y_pred):.2f}")
print("\n📊 تقرير التصنيف:")
print(classification_report(y_test, y_pred))

# تدريب نماذج التنبؤ بعدد عمليات الشراء
X = customer_activity[['total_carts', 'total_views', 'total_spent']]
y = customer_activity['total_purchases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"📊 RMSE لخوارزمية {name}: {rmse:.2f}")
