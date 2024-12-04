import onnxruntime as ort
import numpy as np
import cv2
import time


def detect_qr_code(image):
    session = ort.InferenceSession("qrcode_model.onnx")
    input_shape = session.get_inputs()[0].shape
    print("Model Input Shape:", input_shape)

    # 调整图像大小以匹配模型输入
    target_size = (input_shape[2], input_shape[3])
    inputBlob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, target_size, (0, 0, 0), swapRB=True, crop=False)

    # 确保inputBlob的形状是(1, 3, height, width)
    inputBlob = inputBlob.transpose(0, 3, 1, 2)  # 调整维度顺序

    # 现在inputBlob的形状应该是(1, 3, height, width)
    input_data = inputBlob.reshape(input_shape)

    # 运行模型
    outputs = session.run(None, {session.get_inputs()[0].name: input_data})

    # 获取结果
    output_data = outputs[0]
    prediction = np.argmax(output_data)

    return prediction


# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧（流结束？）")
        break

    # 打印帧的形状以调试
    print("Frame Shape:", frame.shape)

    # 检测二维码
    start_time = time.time()
    prediction = detect_qr_code(frame)
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # 在图像上显示预测结果和FPS
    cv2.putText(frame, f"Class: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('QR Code Detection', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
