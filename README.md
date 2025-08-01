# Phát hiện Làn đường với ENet-LaneNet trên bộ dữ liệu TuSimple

Dự án này triển khai một mô hình học sâu để phát hiện làn đường trong ảnh, dựa trên kiến trúc **LaneNet**. Mô hình được xây dựng bằng PyTorch, sử dụng **ENet** làm kiến trúc nền (backbone) để đảm bảo hiệu quả tính toán. Mục tiêu là thực hiện phân vùng thực thể (instance segmentation) để không chỉ xác định các pixel thuộc về làn đường mà còn phân biệt được các làn đường khác nhau.

Mô hình được huấn luyện trên bộ dữ liệu [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple?select=TUSimple).

## 1. Cơ sở lý thuyết

### 1.1 Kiến trúc LaneNet

LaneNet giải quyết bài toán phát hiện làn đường bằng cách chia thành hai nhiệm vụ nhỏ hơn, được xử lý song song qua một kiến trúc hai nhánh:

#### 🔹 Nhánh Phân vùng Nhị phân (Binary Segmentation Branch)

- **Nhiệm vụ:** Trả lời câu hỏi *"Pixel này có thuộc về một làn đường hay không?"*  
- **Bản chất:** Bài toán phân loại mỗi pixel thành hai lớp: `lane` và `background`.  
- **Đầu ra:** Một bản đồ phân vùng (segmentation map) nơi các pixel thuộc làn đường được đánh dấu.  
- **Hàm mất mát:** `CrossEntropyLoss` được sử dụng để đo lường sai khác giữa dự đoán và nhãn thật.

#### 🔹 Nhánh Nhúng Đặc trưng (Instance Embedding Branch)

- **Nhiệm vụ:** Trả lời câu hỏi *"Pixel này thuộc về làn đường cụ thể nào?"*  
- **Cách hoạt động:** Nhánh này học cách ánh xạ mỗi pixel vào một không gian đặc trưng nhiều chiều (embedding space).  
- **Nguyên tắc:**  
  - Pixel thuộc **cùng một làn đường** → vector embedding gần nhau.  
  - Pixel thuộc **làn đường khác nhau** → vector embedding cách xa nhau.  
- **Hàm mất mát:** `Discriminative Loss`, gồm:
  - **Variance Loss (L_var):** Kéo các embedding cùng làn đường lại gần tâm cụm.
  - **Distance Loss (L_dist):** Đẩy tâm của các cụm làn đường ra xa nhau.
- **Tham số sử dụng:**
  - `δv = 0.5`
  - `δd = 3.0`

---

### 1.2 Hậu xử lý (Post-processing)

Sau khi mô hình sinh ra hai đầu ra từ hai nhánh, cần một bước hậu xử lý để nhóm các pixel thành các làn đường hoàn chỉnh.

- **Phương pháp sử dụng trong project:**  
  Thuật toán phân cụm **DBSCAN** được sử dụng để nhóm các vector embedding (của những pixel được phân loại là lane) lại với nhau.  
  Mỗi cụm kết quả tương ứng với một làn đường được phát hiện.

- **So sánh với paper gốc:**  
  Trong bài báo gốc của LaneNet, một mạng neuron thứ hai là **H-Net** được sử dụng để học phép biến đổi phối cảnh và khớp các pixel thành đường cong bậc 3 (3rd-order polynomial fit).

