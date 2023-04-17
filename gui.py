from PyQt5.QtWidgets import QApplication, QMainWindow,QHBoxLayout, QVBoxLayout,QWidget,QLabel,QPushButton, QFileDialog, QMessageBox, QPlainTextEdit
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
from PIL import Image
from crnn.test_pred import predict
from process_result import get_info

import sys
import os

class OCR_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bank Card / Credi Card OCR System")
        self.resize(800, 600)

        self.yolo = YOLO("./yolo_best.pt")
        self.init_gui()

    def init_gui(self):
        main_widget = QWidget(self)
        main_layout = QVBoxLayout()

        # 图片显示
        self.image_label = QLabel(self)
        main_layout.addWidget(self.image_label)

        # 结果第一栏
        result_line_1 = QHBoxLayout()
        # Card Number标签
        self.card_number = QLabel(self)
        self.card_number.setText("Card Number:")
        result_line_1.addWidget(self.card_number)
        # Card Number结果
        self.card_number_result = QPlainTextEdit()
        self.card_number_result.setReadOnly(True)
        self.card_number_result.setMaximumHeight(35)
        self.card_number_result.setMaximumWidth(250)
        result_line_1.addWidget(self.card_number_result)
        result_line_1.addStretch(1)
        # Valid Date标签
        self.valid_date = QLabel(self)
        self.valid_date.setText("Valid Date:")
        result_line_1.addWidget(self.valid_date)
        # Valid Date结果
        self.valid_date_result = QPlainTextEdit()
        self.valid_date_result.setReadOnly(True)
        self.valid_date_result.setMaximumHeight(35)
        self.valid_date_result.setMaximumWidth(100)
        result_line_1.addWidget(self.valid_date_result)
        main_layout.addLayout(result_line_1)

        # 结果第二栏
        result_line_2 = QHBoxLayout()
        # 银行名称标签
        self.bank_name = QLabel(self)
        self.bank_name.setText("Bank Name:")
        result_line_2.addWidget(self.bank_name)
        # 银行名称结果
        self.bank_name_result = QPlainTextEdit()
        self.bank_name_result.setReadOnly(True)
        self.bank_name_result.setMaximumHeight(35)
        self.bank_name_result.setMaximumWidth(150)
        result_line_2.addWidget(self.bank_name_result)
        result_line_2.addStretch(1)
        # 卡类型标签
        self.card_type = QLabel(self)
        self.card_type.setText("Card Type:")
        result_line_2.addWidget(self.card_type)
        # 卡类型结果
        self.card_type_result = QPlainTextEdit()
        self.card_type_result.setReadOnly(True)
        self.card_type_result.setMaximumHeight(35)
        self.card_type_result.setMaximumWidth(100)
        result_line_2.addWidget(self.card_type_result)
        result_line_2.addStretch(1)
        # 是否是银联卡标签
        self.is_unionpay = QLabel(self)
        self.is_unionpay.setText("Is UnionPay:")
        result_line_2.addWidget(self.is_unionpay)
        # 是否是银联卡结果
        self.is_unionpay_result = QPlainTextEdit()
        self.is_unionpay_result.setReadOnly(True)
        self.is_unionpay_result.setMaximumHeight(35)
        self.is_unionpay_result.setMaximumWidth(100)
        result_line_2.addWidget(self.is_unionpay_result)
        main_layout.addLayout(result_line_2)

        # 按钮
        button_layout = QHBoxLayout()
        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        self.process_button = QPushButton('Process Image', self)
        self.process_button.clicked.connect(self.process_image)
        button_layout.addWidget(self.process_button)

        self.up_button = QPushButton('Up', self)
        self.up_button.clicked.connect(self.image_up)
        button_layout.addWidget(self.up_button)

        self.down_button = QPushButton('Down', self)
        self.down_button.clicked.connect(self.image_down)
        button_layout.addWidget(self.down_button)

        self.save_button = QPushButton('Save', self)
        # self.save_button.clicked.connect(self.image_save)
        button_layout.addWidget(self.save_button)
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_image(self):
        self.clear_result()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")

        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height()))

        self.image_file_path = file_path

    def process_image(self):
        result = self.yolo(self.image_file_path)
        image = Image.open(self.image_file_path)

        res_plotted = result[0].plot()

        boxes = result[0].boxes.xyxy.to('cpu').numpy().astype(int)
        confidences = result[0].boxes.conf.to('cpu').numpy().astype(float)
        labels = result[0].boxes.cls.to('cpu').numpy().astype(int) 

        for box, conf, label in zip(boxes, confidences, labels):
            if conf > 0.5:
                x_min, y_min, x_max, y_max = box
                image_crop = image.crop((x_min,y_min, x_max,y_max))
                result = predict(image_crop)
                print(result)
                result_text = ''
                for i in result[0]:
                    result_text = result_text + i
                # image_crop.convert('RGB').save('card.jpg')
                if label == 0 and conf > 0.7:
                    self.card_number_result.setPlainText(result_text)
                    processed_result = get_info(result_text)
                    self.bank_name_result.setPlainText(processed_result[0])
                    if processed_result[1] == 'DC':
                        self.card_type_result.setPlainText('储蓄卡')
                    elif processed_result[1] == 'CC':
                        self.card_type_result.setPlainText('信用卡')
                    elif processed_result[1] == 'SCC':
                        self.card_type_result.setPlainText('准贷记卡')
                    elif processed_result[1] == 'PC':
                        self.card_type_result.setPlainText('预付费卡')
                    else:
                        self.card_type_result.setPlainText(processed_result[1])
                    self.is_unionpay_result.setPlainText(processed_result[2])
                else:
                    if len(result_text) == 4:
                        result_text = result_text[0:2] + '/' + result_text[2:4]
                    elif len(result_text) == 5:
                        result_text = result_text[0:2] + '/' + result_text[3:5]
                    # elif len(result_text) == 6:
                    #     result_text = result_text[0:2] + '/' + result_text[2:6]
                    elif len(result_text) == 7:
                        result_text = result_text[0:4] + '/' + result_text[5:7]
                    self.valid_date_result.setPlainText(result_text)

        height, width, _ = res_plotted.shape
        bytesPerLine = 3 * width
        self.image_label.setPixmap(QPixmap(QImage(res_plotted.data, width, height, bytesPerLine, QImage.Format_BGR888)).scaled(self.image_label.width(), self.image_label.height()))

    def image_up(self):
        self.clear_result()
        try:
            path = os.path.split(self.image_file_path)
            image_name = os.listdir(path[0])
            n = image_name.index(path[1])
            n = n - 1
            if n < 0:
                n = len(image_name) - 1
            self.image_file_path = path[0] + '/' + image_name[n]
            if image_name[n].split('.')[-1] in ['jpg','png','jpeg']:
                self.process_image()
            else:
                self.image_up()
        except:
            msg = QMessageBox(QMessageBox.Critical, '错误', '未读入图片')
            msg.exec_()

    def image_down(self):
        self.clear_result()
        try:
            path = os.path.split(self.image_file_path)
            image_name = os.listdir(path[0])
            n = image_name.index(path[1])
            n = n + 1
            if n >= len(image_name):
                n = 0
            self.image_file_path = path[0] + '/' + image_name[n]
            if image_name[n].split('.')[-1] in ['jpg','png','jpeg']:
                self.process_image()
            else:
                self.image_down()
        except:
            msg = QMessageBox(QMessageBox.Critical, '错误', '未读入图片')
            msg.exec_()

    def clear_result(self):
        self.card_number_result.setPlainText('')
        self.bank_name_result.setPlainText('')
        self.card_type_result.setPlainText('')
        self.valid_date_result.setPlainText('')
        self.is_unionpay_result.setPlainText('')

if __name__ == "__main__":
    app = QApplication([])
    window = OCR_Window()
    window.show()
    sys.exit(app.exec())