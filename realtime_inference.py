import cv2
import torch
from model import Yolov1
from torchvision import transforms
from utils import *

# 设定设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
def load_model(checkpoint_path):
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# 图像预处理
def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

# 推理并绘制边界框
def predict_and_draw(model, image):
    image_tensor = transform_image(image)
    with torch.no_grad():
        predictions = model(image_tensor)
        bboxes = cellboxes_to_boxes(predictions)
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    return bboxes

# 从视频捕获并进行实时检测
def video_capture_and_detect(model):
    cap = cv2.VideoCapture(0)  # 0 是默认的摄像头
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Detect objects
        bboxes = predict_and_draw(model, frame)
        
        # Draw boxes on the frame
        for box in bboxes:
            class_label = box[0]
            prob_score = box[1]
            x_mid, y_mid, width, height = box[2:6]
            x_mid *= frame.shape[1]
            y_mid *= frame.shape[0]
            width *= frame.shape[1]
            height *= frame.shape[0]
            upper_left_x = int(x_mid - width / 2)
            upper_left_y = int(y_mid - height / 2)

            cv2.rectangle(frame, (upper_left_x, upper_left_y), (upper_left_x + int(width), upper_left_y + int(height)), (0, 255, 0), 2)
            label_text = f"{class_label} ({prob_score:.2f})"
            cv2.putText(frame, label_text, (upper_left_x, upper_left_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 加载模型并启动视频捕捉
model_path = 'res/epoch_208_checkpoint.pth.tar'
model = load_model(model_path)
video_capture_and_detect(model)