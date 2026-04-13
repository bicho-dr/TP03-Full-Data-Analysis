# ==============================
# TP03 - Full Data Analysis Project
# Python + Power BI
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ==============================
# 1️⃣ DATA COLLECTION
# ==============================

data = pd.read_csv(
    r'D:\Big data 02\Tp03\TP03_Full_Data_Analysis\data\data.csv',
    encoding='ISO-8859-1'
)

print("Initial Shape:", data.shape)
print(data.head())

# ==============================
# 2️⃣ FEATURE ENGINEERING
# ==============================

# إنشاء عمود إجمالي المبيعات
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# ==============================
# 3️⃣ DATA CLEANING
# ==============================

# حذف القيم الناقصة
data.dropna(inplace=True)

# حذف المرتجعات والقيم السالبة
data = data[data['Quantity'] > 0]
data = data[data['UnitPrice'] > 0]

print("After Cleaning Shape:", data.shape)

# تحويل التاريخ
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

data['Year'] = data['InvoiceDate'].dt.year
data['Month'] = data['InvoiceDate'].dt.month
data['Day'] = data['InvoiceDate'].dt.day

# ==============================
# 4️⃣ EXPLORATORY DATA ANALYSIS (EDA)
# ==============================

print("\nDescriptive Statistics:\n")
print(data.describe())

# ==============================
# 📊 Monthly Sales Trend
# ==============================

monthly_sales = data.groupby('Month')['TotalPrice'].sum()

plt.figure()
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

# ==============================
# 🌍 Sales by Country
# ==============================

country_sales = data.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)

plt.figure()
country_sales.head(10).plot(kind='bar')
plt.title("Top 10 Countries by Sales")
plt.show()

# ==============================
# 🛍️ Top 10 Products
# ==============================

top_products = data.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False)

plt.figure()
top_products.head(10).plot(kind='bar')
plt.title("Top 10 Products by Sales")
plt.show()

# ==============================
# 👥 Top 10 Customers
# ==============================

top_customers = data.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)

plt.figure()
top_customers.head(10).plot(kind='bar')
plt.title("Top 10 Customers")
plt.show()

# ==============================
# 5️⃣ CUSTOMER SEGMENTATION (CLUSTERING)
# ==============================

# تجميع البيانات حسب العميل
customer_data = data.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'Quantity': 'sum'
}).reset_index()

# تطبيق KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['TotalPrice', 'Quantity']])

print("\nCluster Distribution:\n")
print(customer_data['Cluster'].value_counts())

# رسم النتائج
plt.figure()
sns.scatterplot(
    x='TotalPrice',
    y='Quantity',
    hue='Cluster',
    data=customer_data
)
plt.title("Customer Segmentation")
plt.show()

# ==============================
# 6️⃣ SAVE CLEANED DATA FOR POWER BI
# ==============================

data.to_csv(
    r'D:\Big data 02\Tp03\TP03_Full_Data_Analysis\outputs\cleaned_ecommerce_data.csv',
    index=False
)

print("\n✅ Project Completed Successfully!")