# Phát hiện Làn đường với ENet-LaneNet trên bộ dữ liệu TuSimple

Dự án này triển khai một mô hình học sâu để phát hiện làn đường trong ảnh, dựa trên kiến trúc **LaneNet**. Mô hình được xây dựng bằng PyTorch, sử dụng **ENet** làm kiến trúc nền (backbone) để đảm bảo hiệu quả tính toán. Mục tiêu là thực hiện phân vùng thực thể (instance segmentation) để không chỉ xác định các pixel thuộc về làn đường mà còn phân biệt được các làn đường khác nhau.

Mô hình được huấn luyện trên bộ dữ liệu [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple?select=TUSimple).

.1. Kiến trúc LaneNet
LaneNet giải quyết bài toán phát hiện làn đường bằng cách chia nó thành hai nhiệm vụ nhỏ hơn và xử lý song song thông qua một kiến trúc hai nhánh:

Nhánh Phân vùng Nhị phân (Binary Segmentation Branch):

Nhiệm vụ: Trả lời câu hỏi "Pixel này có thuộc về một làn đường hay không?". Đây là một bài toán phân loại pixel thành hai lớp: lane và background.

Đầu ra: Một bản đồ phân vùng (segmentation map) nơi các pixel thuộc làn đường được đánh dấu.

Hàm mất mát: CrossEntropyLoss được sử dụng để đo lường sự khác biệt giữa dự đoán và nhãn phân vùng thực tế.

Nhánh Nhúng Đặc trưng (Instance Embedding Branch):

Nhiệm vụ: Trả lời câu hỏi "Pixel này thuộc về làn đường cụ thể nào?". Nhánh này học cách ánh xạ mỗi pixel của làn đường vào một không gian đặc trưng (embedding space) nhiều chiều.

Nguyên tắc: Các pixel thuộc cùng một làn đường sẽ có vector embedding gần nhau, trong khi các pixel thuộc các làn đường khác nhau sẽ có vector embedding ở xa nhau.

Hàm mất mát: Discriminative Loss được sử dụng để tối ưu hóa không gian embedding này. Hàm loss này bao gồm:

Variance Loss (L 
var
​
 ): Kéo các embedding của cùng một làn đường lại gần tâm của chúng.

Distance Loss (L 
dist
​
 ): Đẩy tâm của các làn đường khác nhau ra xa nhau.

Trong project này, các tham số của DiscriminativeLoss được thiết lập là δ 
v
​
 =0.5 và δ 
d
​
 =3.

1.2. Hậu xử lý (Post-processing)
Sau khi mô hình đưa ra dự đoán từ hai nhánh, cần một bước hậu xử lý để nhóm các pixel lại thành các làn đường hoàn chỉnh.

Phương pháp sử dụng: Trong project này, thuật toán phân cụm DBSCAN được sử dụng. Thuật toán này sẽ nhóm các vector embedding (từ các pixel được dự đoán là lane) lại với nhau. Mỗi cụm kết quả tương ứng với một làn đường được phát hiện.

Phương pháp trong paper gốc: Paper gốc của LaneNet sử dụng một mạng neuron thứ hai là H-Net để học phép biến đổi phối cảnh và thực hiện "fit" các điểm ảnh thành một đường cong đa thức bậc 3 (3rd-order polynomial fit).
