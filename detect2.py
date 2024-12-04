import onnxruntime as ort
import numpy as np
import cv2
import time


def detect_qr_code(image):
    session = ort.InferenceSession("qrcode_model.onnx")
    input_shape = session.get_inputs()[0].shape
    # print("Model Input Shape:", input_shape)  # 注释掉以避免重复输出

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

    outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    output_data = outputs[0]
    prediction = np.argmax(output_data)

    # 输出模型对每个类别的预测概率
    print(f"Prediction Probabilities: {output_data[0]}")

    return prediction


# 初始化视频捕获
video_path = '20241026_231031.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        print("视频结束")
        break

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
    if cv2.waitKey(30) & 0xFF == ord('q'):  # 增加了waitKey的延迟时间以适应视频播放速度
        break

# 释放视频捕获和关闭窗口
cap.release()
cv2.destroyAllWindows()
