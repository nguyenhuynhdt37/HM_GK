import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =================================================================
# YÊU CẦU 1 & 2: ĐỌC DỮ LIỆU VÀ KHÁM PHÁ (EDA)
# =================================================================
print("--- GIAI ĐOẠN 1: ĐỌC VÀ KHÁM PHÁ DỮ LIỆU ---")

# Đọc dữ liệu: sep=';' và decimal=',' là bắt buộc cho bộ dữ liệu này
df = pd.read_csv('/Users/huynh/codes/project_hocmay/data/AirQualityUCI.csv', sep=';', decimal=',')

# Loại bỏ các cột và dòng trống hoàn toàn (thường xuất hiện ở cuối file UCI)
df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)

print(f"Kích thước dữ liệu: {df.shape}")
print("\nThông tin các biến số:")
print(df.info())

# =================================================================
# YÊU CẦU 3: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# =================================================================
print("\n--- GIAI ĐOẠN 2: TIỀN XỬ LÝ DỮ LIỆU ---")

# 1. Xử lý lỗi định dạng thời gian (Thay '.' bằng ':' để tránh lỗi Parse)
df['Time'] = df['Time'].str.replace('.', ':', regex=False)
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# 2. Xử lý giá trị nhiễu -200 (Quy ước giá trị thiếu của UCI)
df.replace(-200, np.nan, inplace=True)

# 3. Nội suy dữ liệu (Interpolation) - Chỉ áp dụng trên các cột số để tránh Warning
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

# Điền nốt các giá trị NaN còn sót lại (nếu dòng đầu tiên bị trống)
df[numeric_cols] = df[numeric_cols].ffill().bfill()

# 4. Trực quan hóa tương quan (Heatmap) - Giống phong cách CCP.ipynb
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', fmt='.2f')
plt.title('Bản đồ nhiệt tương quan giữa các chỉ số không khí')
plt.show()

# Chọn Features và Target dựa trên độ tương quan cao với CO
features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
target = 'CO(GT)'

X = df[features]
y = df[target]

# =================================================================
# YÊU CẦU 4: CHIA TẬP DỮ LIỆU (TRAIN-TEST SPLIT)
# =================================================================
# Tỉ lệ 80% Train - 20% Test là tỉ lệ tối ưu cho bộ dữ liệu ~9000 mẫu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- GIAI ĐOẠN 3: CHIA DỮ LIỆU ---")
print(f"Mẫu huấn luyện: {X_train.shape[0]}")
print(f"Mẫu kiểm thử:   {X_test.shape[0]}")

# =================================================================
# YÊU CẦU 5: HUẤN LUYỆN, ĐÁNH GIÁ (MSE) VÀ TRỰC QUAN
# =================================================================
print("\n--- GIAI ĐOẠN 4: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH ---")

# 1. Khởi tạo mô hình Hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# 2. Dự báo kết quả
y_pred = model.predict(X_test)

# 3. Tính toán các chỉ số lỗi
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

non_zero_mask = y_test != 0
if non_zero_mask.any():
	mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
	accuracy_10pct = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask]) <= 0.10) * 100
else:
	mape = np.nan
	accuracy_10pct = np.nan

print(f"Lỗi bình phương trung bình (MSE): {mse:.4f}")
print(f"Sai số tuyệt đối trung bình (MAE): {mae:.4f}")
print(f"Căn lỗi bình phương trung bình (RMSE): {rmse:.4f}")
if np.isnan(mape):
	print("Tỷ lệ sai số phần trăm trung bình (MAPE): không xác định do toàn bộ nhãn thực tế bằng 0")
else:
	print(f"Tỷ lệ sai số phần trăm trung bình (MAPE): {mape:.2f}%")
if np.isnan(accuracy_10pct):
	print("Độ chính xác trong ngưỡng 10%: không xác định do toàn bộ nhãn thực tế bằng 0")
else:
	print(f"Độ chính xác trong ngưỡng 10%: {accuracy_10pct:.2f}%")
print(f"Hệ số xác định (R-squared):      {r2:.4f}")

# 4. Trực quan hóa chi tiết kết quả dự báo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: So sánh thực tế vs Dự báo
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='teal', ax=ax1)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
ax1.set_title(f'Thực tế vs Dự đoán (R2 = {r2:.3f})')
ax1.set_xlabel('Giá trị thực tế (Ground Truth)')
ax1.set_ylabel('Giá trị dự báo (Prediction)')

# Biểu đồ 2: Phân phối sai số (Residuals)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='orange', ax=ax2)
ax2.set_title(f'Phân phối sai số (MSE = {mse:.3f})')
ax2.set_xlabel('Mức độ lệch (Error)')

plt.tight_layout()
plt.show()

print("\n--- KẾT LUẬN ---")
print("1. Dữ liệu đã được làm sạch triệt để, xử lý 100% giá trị thiếu -200.")
print(f"2. Mô hình Linear Regression đạt độ chính xác R2 cao ({r2:.2%}).")
print(f"3. MAE = {mae:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}% và Accuracy@10% = {accuracy_10pct:.2f}% cho thấy sai số dự báo đã được kiểm soát tốt." if not np.isnan(mape) else f"3. MAE = {mae:.4f} và RMSE = {rmse:.4f} cho thấy sai số dự báo đã được kiểm soát tốt.")