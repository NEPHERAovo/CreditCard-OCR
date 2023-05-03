import cv2
import numpy as np
import sys

def rectify(img_src): 
    src = cv2.imread(img_src)
    cv2.imshow("src img", src)

    source = src.copy()
    bkup = src.copy()
    img = src.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转为灰度图
    cv2.imshow("gray", gray)

    img = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义卷积核

    img = cv2.dilate(img, element, iterations=1)  # 膨胀
    cv2.imshow("dilate", img)

    img = cv2.Canny(img, 30, 180, 3)  # 边缘提取
    cv2.imshow("get contour", img)

    img = cv2.dilate(img, element, iterations=1)  # 膨胀
    cv2.imshow("dilated", img)

    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for i in range(len(contours)):
        tmparea = cv2.contourArea(contours[i])
        if tmparea > max_area:
            index = i
            max_area = tmparea
    max_contour = [contours[index]]

    mask = np.zeros_like(src)
    cv2.drawContours(mask, max_contour, 0, 255, -1) 
    cv2.imshow("max area", mask)


    lines = cv2.HoughLines(mask, 1, np.pi/180, 160)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow("lines", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    img_src = './datasets/everything/17.jpg'
    img_rectified = rectify(img_src)
    # cv2.imshow('rectified', img_rectified)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()