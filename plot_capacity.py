import pandas as pd
import matplotlib.pyplot as plt
import os

# Định nghĩa đường dẫn
data_path = "/Users/huynh/codes/project_hocmay/data/metadata.csv"
output_path = "/Users/huynh/.gemini/antigravity/brain/8dda05c3-0018-4213-b7a2-cf8c97de54ca/artifacts/b0005_degradation.png"

# Đọc file metadata.csv
df = pd.read_csv(data_path)

# Lọc dữ liệu: Chỉ lấy các chu kỳ xả (discharge) của pin B0005
df_b0005 = df[(df['battery_id'] == 'B0005') & (df['type'] == 'discharge')].copy()

# Xoá các dòng không có giá trị Capacity (nếu có)
df_b0005 = df_b0005.dropna(subset=['Capacity'])

# Tạo trục X là số chu kỳ (Cycle Index)
df_b0005['cycle_index'] = range(1, len(df_b0005) + 1)

# Bắt đầu vẽ biểu đồ
plt.figure(figsize=(10, 6))

# Đường biểu diễn sự suy thoái dung lượng thực tế
plt.plot(df_b0005['cycle_index'], df_b0005['Capacity'], marker='o', linestyle='-', color='#1f77b4', markersize=4, label='Dung lượng thực (Capacity)')

# Vẽ đường gạch ngang màu đỏ thể hiện ngưỡng "End of Life" (EOL) - Pin hỏng (1.4 Ah theo README)
plt.axhline(y=1.4, color='red', linestyle='--', linewidth=2, label='Ngưỡng hỏng (End of Life - 1.4 Ahr)')

# Nhãn và trang trí
plt.title('Sự Lụi Tàn Của Pin NASA B0005 (Dự đoán Tương lai / RUL)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Số chu kỳ Xả (Discharge Cycles)', fontsize=12)
plt.ylabel('Dung lượng tối đa chứa được (Ahr)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')

# Làm mượt các viền xung quanh
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Lưu thành file ảnh
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ thành công tại: {output_path}")
