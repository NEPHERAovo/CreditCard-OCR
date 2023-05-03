import cv2
import numpy as np

def reorder_points(points):
    # 按照 x 坐标排序
    sorted_x = sorted(points, key=lambda x: x[0])
    # 将结果拆分成两行
    row1 = sorted_x[:2]
    row2 = sorted_x[2:]
    # 对每一行按照 y 坐标排序
    row1 = sorted(row1, key=lambda x: x[1])
    row2 = sorted(row2, key=lambda x: x[1])
    # 合并结果
    bottom_left = row1[1]
    bottom_right = row2[1]
    top_right = row2[0]
    top_left = row1[0]

    # 按照正确的顺序排列四个点
    return np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)

def rectify(img_src): 
    image = cv2.imread(img_src)
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 30, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤掉面积过小的轮廓
    contours = [c for c in contours if cv2.contourArea(c) > 5]
    # 创建一个空白图像，大小与原始图像相同
    mask = np.zeros(gray.shape, dtype=np.uint8)

    # 在空白图像上绘制筛选后的轮廓
    cv2.drawContours(mask, contours, -1, 255, -1)

    # 使用闭操作连接非闭合区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('a',closed)
    # 找到轮廓
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    # 近似获取轮廓的顶点
    epsilon = 0.1 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    if len(approx) != 4:
        while len(approx) < 4:
            epsilon *= 0.99
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
        while len(approx) > 4:
            epsilon *= 1.01
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
    src = reorder_points(approx.reshape(4, 2))
    # 计算出变换矩阵
    w1, h1 = 640, 400
    w2, h2 = 250, 350
    dst = np.float32([[0, h1], [w1, h1], [w1, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(src, dst)

    # 进行透视变换
    warped = cv2.warpPerspective(image, M, (w1, h1))
    
    # 绘制轮廓和顶点
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    for point in approx:
        cv2.circle(image, tuple(point[0]), 5, (0, 0, 255), -1)


    # # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Warped Image', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warped


if __name__ == "__main__":
    img_src = './test/163.jpg'
    img_rectified = rectify(img_src)
    # cv2.imshow('rectified', img_rectified)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()