import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import DBSCAN 

from lane_detector import ENet

# Load the pre-trained model
model_path = '/home/loipham/INTERN/kaggle/models/ENET_epoch_20.pth'
enet_model = ENet(binary_seg=2, embedding_dim=4)  

# Load the trained model's weights
enet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
enet_model.eval()

def process_and_visualize(input_image_path):
    input_image_bgr = cv2.imread(input_image_path)
    input_image_resized_bgr = cv2.resize(input_image_bgr, (512, 256))
    input_image_rgb = cv2.cvtColor(input_image_resized_bgr, cv2.COLOR_BGR2RGB)

    input_image_gray = cv2.cvtColor(input_image_resized_bgr, cv2.COLOR_BGR2GRAY)
    input_image_gray_expanded = input_image_gray[..., None]
    input_tensor = torch.from_numpy(input_image_gray_expanded).float().permute(2, 0, 1)

    with torch.no_grad():
        binary_logits, instance_logits = enet_model(input_tensor.unsqueeze(0))

    binary_seg = torch.argmax(binary_logits, dim=1).squeeze().numpy().astype(np.uint8)

    embedding = instance_logits.squeeze().cpu().numpy()

    embedding_pixels = embedding.transpose(1, 2, 0)[binary_seg == 1]

    overlay_image = input_image_rgb.copy()

    if embedding_pixels.shape[0] > 100:
        # Phân cụm các pixel bằng DBSCAN
        dbscan = DBSCAN(eps=0.4, min_samples=100)
        labels = dbscan.fit_predict(embedding_pixels)

        lane_coords = np.where(binary_seg == 1)
        lane_coords = np.vstack((lane_coords[0], lane_coords[1])).T

        # 5. Định nghĩa màu và vẽ từng làn đường
        colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)] # Đỏ, Vàng, Xanh lá, Xanh dương, Tím
        unique_labels = np.unique(labels[labels != -1]) # Bỏ qua nhiễu

        for i, label in enumerate(unique_labels):
            color = colors[i % len(colors)]
            # Lấy tọa độ của các pixel trong cụm hiện tại
            current_lane_pixels = lane_coords[labels == label]
            # Vẽ các pixel này lên ảnh overlay
            for pixel in current_lane_pixels:
                # Tọa độ trong numpy là (hàng, cột), cần đảo ngược cho cv2 (x, y)
                overlay_image[pixel[0], pixel[1]] = color
    else:
        overlay_image[binary_seg == 1] = [255, 0, 0]

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image_rgb)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_image)
    plt.title('Output')
    plt.axis('off')

input_image_path = '/home/loipham/INTERN/road3.jpg'
process_and_visualize(input_image_path)

plt.show()