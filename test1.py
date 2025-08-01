import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from lane_detector import ENet

# Load the pre-trained model
model_path = '/home/loipham/INTERN/kaggle/models/ENET_epoch_20.pth' 
enet_model = ENet(2, 4)  # Assuming you used the same model architecture

# Load the trained model's weights
enet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
enet_model.eval()  

# Define a function to process and visualize the output
def process_and_visualize(input_image_path):
    # Load ảnh bằng OpenCV (định dạng BGR)
    input_image_bgr = cv2.imread(input_image_path)
    # Resize ảnh về kích thước input của model
    input_image_resized_bgr = cv2.resize(input_image_bgr, (512, 256))

    # Chuyển đổi từ BGR (OpenCV) sang RGB (Matplotlib) để màu sắc hiển thị đúng
    # Đây là ảnh gốc sẽ được hiển thị bên trái
    input_image_rgb = cv2.cvtColor(input_image_resized_bgr, cv2.COLOR_BGR2RGB)

    # Chuẩn bị ảnh cho model (chuyển sang ảnh xám)
    input_image_gray = cv2.cvtColor(input_image_resized_bgr, cv2.COLOR_BGR2GRAY)
    input_image_gray = input_image_gray[..., None]
    input_tensor = torch.from_numpy(input_image_gray).float().permute(2, 0, 1)

    # Đưa ảnh qua model
    with torch.no_grad():
        binary_logits, instance_logits = enet_model(input_tensor.unsqueeze(0))

    # Lấy kết quả binary segmentation mask
    binary_seg = torch.argmax(binary_logits, dim=1).squeeze().numpy().astype(np.uint8)

    # Tạo ảnh overlay
    # Tạo một bản sao của ảnh màu để vẽ lên
    overlay_image = input_image_rgb.copy()
    # Dùng mask để tìm các pixel của làn đường và tô màu đỏ đậm [255, 0, 0]
    overlay_image[binary_seg == 1] = [255, 0, 0] # Tô màu đỏ

    # Hiển thị kết quả
    plt.figure(figsize=(15, 6))

    # Plot ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(input_image_rgb)
    plt.title('Input')
    plt.axis('off')

    # Plot ảnh đã tô màu làn đường
    plt.subplot(1, 2, 2)
    plt.imshow(overlay_image)
    plt.title('Output')
    plt.axis('off')

# Replace 'input_image.jpg' with the path to your test image
input_image_path = '/home/loipham/INTERN/road2.jpg'
process_and_visualize(input_image_path)

plt.show()
