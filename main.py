# test.ipynb  

# 导入必要的库  
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 开始时间
start_time = time.time()


# 第一步：图像预处理
def preprocess_image(image_path, resize_width=1000):
    print(f"加载图像: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return None, None
        # 获取原始尺寸
    height, width = img.shape[:2]
    # 计算缩放比例
    if width > resize_width:
        scale_ratio = resize_width / width
        new_size = (resize_width, int(height * scale_ratio))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        print(f"图像 {image_path} 缩放到 {new_size}")
    else:
        print(f"图像 {image_path} 尺寸无需缩放")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用高斯滤波
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    print(f"图像 {image_path} 预处理完成")
    return img, gray


img1_path = '8.jpg'
img2_path = '9.jpg'

img1_color, img1_gray = preprocess_image(img1_path)
img2_color, img2_gray = preprocess_image(img2_path)

if img1_gray is None or img2_gray is None:
    print("其中一张图像加载失败，请检查图像路径或图像文件。")
    exit()

# 预处理后的灰度图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('预处理后的图像1 灰度图')
plt.imshow(img1_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('预处理后的图像2 灰度图')
plt.imshow(img2_gray, cmap='gray')
plt.axis('off')
plt.show()


# 第二步：特征点和特征描述符提取
def extract_features_orb(gray_image, image_path):
    # 使用 ORB 检测器
    orb = cv2.ORB_create(nfeatures=2500,scaleFactor=1.2)  # 设置特征点数量上限
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    if descriptors is None:
        print(f"{image_path} 无法提取 ORB 特征描述符。")
        return keypoints, None
    print(f"{image_path} ORB 特征点数量: {len(keypoints)}")
    return keypoints, descriptors


kp1, des1 = extract_features_orb(img1_gray, img1_path)
kp2, des2 = extract_features_orb(img2_gray, img2_path)

if des1 is None or des2 is None:
    print("其中一张图像的描述符为空，无法进行匹配。请尝试使用包含更多特征的图像。")
    exit()


# 绘制特征点
def draw_keypoints(img, keypoints, title):
    img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


draw_keypoints(img1_color, kp1, '图像1 的特征点')
draw_keypoints(img2_color, kp2, '图像2 的特征点')


# 第三步：特征点匹配
def match_features_orb(des1, des2, image1, image2):
    # 使用BFMatcher进行特征匹配，使用Hamming距离
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        print(f"特征匹配时出错: {e}")
        return []

        # 使用 Lowe 的比率测试筛选匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"良好匹配点数量: {len(good_matches)}")
    return good_matches


good_matches = match_features_orb(des1, des2, img1_path, img2_path)

if len(good_matches) < 4:
    print("良好匹配点数量不足 (<4)，无法计算单应性矩阵。请尝试使用包含更多特征的图像。")
    exit()


# 绘制匹配结果
def draw_matches(img1, kp1, img2, kp2, matches):
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.title('特征点匹配结果')
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


draw_matches(img1_color, kp1, img2_color, kp2, good_matches)


# 第四步：计算单应性矩阵并进行图像拼接
def stitch_images(img1, img2, kp1, kp2, matches):
    if len(matches) < 4:
        print("匹配点数量不足，无法计算单应性矩阵。")
        return None

        # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("无法计算单应性矩阵。")
        return None
    print("单应性矩阵计算完成")

    # 获取拼接后的图像尺寸
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # 计算目标图像的尺寸
    corners_img1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    corners_img1_transformed = cv2.perspectiveTransform(corners_img1, H)
    corners_img2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_img1_transformed, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation_dist = [-xmin, -ymin]

    # 平移单应性矩阵以包含所有图像
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])

    # 进行透视变换和图像拼接
    stitched_img = cv2.warpPerspective(img1, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    stitched_img[translation_dist[1]:height2 + translation_dist[1],
    translation_dist[0]:width2 + translation_dist[0]] = img2

    print("图像拼接完成")
    return stitched_img


stitched_image = stitch_images(img1_color, img2_color, kp1, kp2, good_matches)

if stitched_image is not None:
    # 显示拼接后的图像
    plt.figure(figsize=(20, 10))
    plt.title('拼接后的图像')
    plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


    # ------------ 后处理步骤：图像融合和裁剪 ------------

    # 图像融合：多频带融合（可选）
    def blend_images(stitched_img, img2):
        # 这个函数可以使用更高级的融合技术，比如多频带融合
        # 但是为了简便，这里使用线性渐变融合
        # 根据您的需求，可以拓展为更复杂的融合算法
        # 此处留作扩展
        return stitched_img  # 目前不进行额外融合


    blended_image = blend_images(stitched_image, img2_color)


    # 自动裁剪黑色边缘
    def crop_black_borders(image):
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值化
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # 找到所有的非零点坐标
        coords = cv2.findNonZero(thresh)
        if coords is None:
            print("无法找到非黑色区域，返回原图。")
            return image
            # 找到最小矩形
        x, y, w, h = cv2.boundingRect(coords)
        # 裁剪图像
        cropped = image[y:y + h, x:x + w]
        print(f"图像自动裁剪: (x={x}, y={y}, w={w}, h={h})")
        return cropped


    final_image = crop_black_borders(blended_image)

    # 显示裁剪后的最终图像
    plt.figure(figsize=(20, 10))
    plt.title('裁剪后的最终拼接图像')
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # ------------ 后处理结束 ------------

    # 记录处理结束时间
    end_time = time.time()
    total_time = end_time - start_time
    frame_rate = 1 / total_time if total_time > 0 else float('inf')

    print(f"处理时间: {total_time:.4f} 秒")
    print(f"帧率: {frame_rate:.2f} 帧/秒")
else:
    print("图像拼接失败。")