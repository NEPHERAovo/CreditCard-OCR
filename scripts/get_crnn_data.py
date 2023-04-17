import os
import numpy as np
import cv2
import random

SOURCE_PATH = 'D:\Softwares\Python\CreditCard-OCR\datasets/recognition/train_images/'
DESTINATION_PATH = 'D:\Softwares\Python\CreditCard-OCR\datasets/recognition/processed/'


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def rotate(img):
    angle = random.randint(-10, 10)
    (h,w) = img.shape[:2]
    (cx,cy) = (w/2,h/2)
    
    #设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    # 计算图像旋转后的新边界
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    
    return cv2.warpAffine(img,M,(nW,nH))

def rand_resize(img,jitter=.3):
    w, h, _ = img.shape
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    new_img = cv2.resize(img, (nh, nw), cv2.INTER_CUBIC)
    return new_img


def place_img(img):
    H, W, C = img.shape
    offestH = int(rand(int(H * 0.02), int(H * 0.04)))
    offestW = int(rand(int(W * 0.02), int(W * 0.04)))
    dst = np.zeros((H, W, C), np.uint8)
    dst1 = np.zeros((H, W, C), np.uint8)
    dst2 = np.zeros((H, W, C), np.uint8)
    dst3 = np.zeros((H, W, C), np.uint8)

    for i in range(H - offestH):
        for j in range(W - offestW):
            dst[i + offestH, j + offestW] = img[i, j]
            dst1[i, j] = img[i + offestH, j + offestW]
            dst2[i + offestH, j] = img[i, j + offestW]
            dst3[i, j + offestW] = img[i + offestH, j]

    result = [dst, dst1, dst2, dst3]
    return result[random.randint(0, 3)]


def colormap(img):
    rand_b = rand() + 0.5
    rand_g = rand() + 0.5
    rand_r = rand() + 0.5
    H, W, C = img.shape

    dst = np.zeros((H, W, C), np.uint8)
    for i in range(H):
        for j in range(W):
            (b, g, r) = img[i, j]
            b = int(b * rand_b)
            g = int(g * rand_g)
            r = int(r * rand_r)
            if b > 255:
                b = 255
            if g > 255:
                g = 255
            if r > 255:
                r = 255
            dst[i][j] = (b, g, r)

    return dst


# def blur(img):
#     img_GaussianBlur = cv2.GaussianBlur(img, (5, 5), 0)
#     img_Mean = cv2.blur(img, (5, 5))
#     img_Median = cv2.medianBlur(img, 3)
#     img_Bilater = cv2.bilateralFilter(img, 5, 100, 100)

#     result = [img_GaussianBlur, img_Mean, img_Median, img_Bilater]

#     return result[random.randint(0, 3)]


def noise(img):
    H, W, C = img.shape
    noise_img = np.zeros((H, W, C), np.uint8)

    for i in range(H):
        for j in range(W):
            noise_img[i, j] = img[i, j]

    for i in range(255):
        x = np.random.randint(H)
        y = np.random.randint(W)
        noise_img[x, y, :] = 255

    return noise_img


def process(img_list, amount):
    for i in range(amount):
        img_numbers = random.randint(4, 6)
        chosen_filenames = random.sample(img_list, img_numbers)
        
        dst = []
        file_name = ''
        for img_name in chosen_filenames:
            file_name += img_name.split('.')[0][0:4]
            img = cv2.imread(SOURCE_PATH + img_name,-1)

            if random.randint(0, 1) == 0:
                img = noise(img)
            if random.randint(0, 1) == 0:
                img = rand_resize(img)
            if random.randint(0, 1) == 0:
                img = colormap(img)
            # if random.randint(0, 1) == 0:
            #     img = blur(img)
            if random.randint(0, 1) == 0:
                i = random.randint(0, 4)
                for j in range(i):
                    img = place_img(img)
            
            if random.randint(0, 1) == 0:
                img = rotate(img)

            dst.append(img)

        heights = [image.shape[0] for image in dst]
        max_height = max(heights)
        resized_images = [cv2.resize(image, (int(image.shape[1] * max_height / image.shape[0]), max_height)) for image in dst]

        dst = np.hstack(resized_images)
        cv2.imwrite(DESTINATION_PATH + file_name + '-' +str(i) + '.png', dst)


if __name__ == '__main__':
    trainval_percent = 0.2
    train_percent = 0.8
    img_list = os.listdir(SOURCE_PATH)

    process(img_list, 20000)
