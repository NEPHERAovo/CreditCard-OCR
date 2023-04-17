import os
import cv2
 
def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if "txt" in filename:
                Filelist.append(os.path.join(home, filename))
    return Filelist
 
 
if __name__ == "__main__":
    picPath = 'D:\Softwares\Python\CreditCard-OCR\scripts/image'
    filePath = 'D:\Softwares\Python\CreditCard-OCR\scripts/txt'
    outputPath = 'D:\Softwares\Python\CreditCard-OCR\scripts/text_dst'
    Filelist = get_filelist(filePath)
    print(len(Filelist))
    for filename in Filelist:
        output_path = filename.replace(filePath, outputPath)
        outputdir = output_path.rsplit('\\', 1)[0]
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        file_lineinfo = open(output_path, 'w', encoding='utf-8')
 
        f = open(filename, encoding='utf-8-sig')
        img = cv2.imread(picPath +'\\'+ filename.split('\\')[-1].replace('txt', 'jpg'))
        if img is None:
            img = cv2.imread(picPath +'\\'+ filename.split('\\')[-1].replace('txt', 'png'))

        for line in f.readlines():
            data = line.replace('\n', '')
            numbers = line.split(',')
            for i in range(len(numbers)-1):
                numbers[i] = int(numbers[i])
            x1 = min(numbers[0],numbers[2],numbers[4],numbers[6])
            x2 = max(numbers[0],numbers[2],numbers[4],numbers[6])
            y1 = min(numbers[1],numbers[3],numbers[5],numbers[7])
            y2 = max(numbers[1],numbers[3],numbers[5],numbers[7])
            
            w = x2-x1
            h = y2-y1
            x = x1+w/2
            y = y1+h/2

            w = str(w/img.shape[1])
            h = str(h/img.shape[0])
            x = str(x/img.shape[1])
            y = str(y/img.shape[0])

            file_lineinfo.write('0'+' '+ str(x)+ ' '+ str(y)+ ' '+ w+ ' '+ h+'\n')
        f.close()
        file_lineinfo.close()
