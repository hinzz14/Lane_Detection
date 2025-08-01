# Phát hiện Làn đường với ENet-LaneNet trên bộ dữ liệu TuSimple

Dự án này triển khai một mô hình học sâu để phát hiện làn đường trong ảnh, dựa trên kiến trúc **LaneNet**. Mô hình được xây dựng bằng PyTorch, sử dụng **ENet** làm kiến trúc nền (backbone) để đảm bảo hiệu quả tính toán. Mục tiêu là thực hiện phân vùng thực thể (instance segmentation) để không chỉ xác định các pixel thuộc về làn đường mà còn phân biệt được các làn đường khác nhau.

Mô hình được huấn luyện trên bộ dữ liệu [TuSimple Lane Detection Challenge](https://www.kaggle.com/datasets/manideep1108/tusimple?select=TUSimple).
