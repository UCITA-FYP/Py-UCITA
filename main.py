from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
import pytesseract
import argparse
from googletrans import Translator
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageDraw, ImageFont
from PIL import Image as ImagePIL
import numpy as np
import cv2
from kivy.uix.filechooser import FileChooserListView, FileChooserIconView
from kivy.uix.label import Label

# Py Tesseract path
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

# Default choosing for Page segmentation modes and ocr engine
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--psm", type=int, default=6,
                help="Tesseract PSM mode")
ap.add_argument("-", "--oem", type=int, default=3,
                help="Tesseract OEM mode")
args = vars(ap.parse_args())


class UCITA(App):

    def build(self):
        # Title
        self.title = 'Welcome to UCITA'

        # FloatLayout display
        layout = FloatLayout()
        self.label = Label(text="USEK Instant Translator App", size_hint=(0, None), font_size=60, pos_hint={'x': .5,
                                                                                                            'y': .9},
                           color=(0, 0, 1, 1))
        layout.add_widget(self.label)

        # Widget of the video and image display
        self.img1 = Image(source="Main.png", pos_hint={'x': 0.20, 'y': 0.32}, size_hint=(0.6, 0.6))

        # Access to the Lap CAM
        self.capture = cv2.VideoCapture(0)
        layout.add_widget(self.img1)

        # Translate language

        self.label = Label(text="Translate from:", size_hint=(0, None), font_size=30, pos_hint={'x': 0.29, 'y': 0.2},
                           color=(0, 0, 1, 1))
        layout.add_widget(self.label)
        self.label = Label(text="Translate to:", size_hint=(0, None), font_size=30, pos_hint={'x': 0.6, 'y': 0.2},
                           color=(0, 0, 1, 1))
        layout.add_widget(self.label)

        dropdown = DropDown()
        ArrayLang = ['afrikaans', 'amharic', 'arabic', 'azerbaijani', 'belarusian', 'bengali', 'bosnian',
                     'bulgarian',
                     'catalan', 'cebuano', 'chinese (simplified)', 'chinese (traditional)', 'croatian', 'czech',
                     'danish', 'dutch', 'english', 'esperanto', 'estonian', 'french', 'finnish', 'galician',
                     'georgian',
                     'german', 'hebrew', 'hindi', 'hungarian', 'italian', 'japanese', 'javanese', 'kannada',
                     'kazakh',
                     'korean', 'khmer', 'kurdish', 'lao', 'latin', 'latvian', 'lithuanian', 'macedonian', 'malay',
                     'malayalam', 'maltese', 'marathi', 'myanmar', 'nepali', 'norwegian', 'pashto', 'polish',
                     'portuguese', 'romanian', 'russian', 'serbian', 'sinhala', 'slovak', 'slovenian', 'spanish',
                     'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'ukrainian', 'urdu',
                     'uzbek',
                     'vietnamese', 'yiddish']
        for lang in ArrayLang:
            btnf = Button(text=lang, size_hint_y=None, height=40)

            # binding the button to show the text when selected
            btnf.bind(on_release=lambda btnf: dropdown.select(btnf.text))
            # then add the button inside the dropdown
            dropdown.add_widget(btnf)

        self.btn4 = Button(text='', font_size="20", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1),
                           size_hint=(.15, .09), pos_hint={'x': 0.3, 'y': 0.15})

        self.btn4.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: setattr(self.btn4, 'text', x))
        layout.add_widget(self.btn4)

        dropdown2 = DropDown()
        for lang in ArrayLang:
            btnt = Button(text=lang, size_hint_y=None, height=40)

            # binding the button to show the text when selected
            btnt.bind(on_release=lambda btnt: dropdown2.select(btnt.text))
            # then add the button inside the dropdown
            dropdown2.add_widget(btnt)

        self.btn5 = Button(text='', font_size="20", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1),
                           size_hint=(.15, .09), pos_hint={'x': .6, 'y': 0.15})

        self.btn5.bind(on_release=dropdown2.open)
        dropdown2.bind(on_select=lambda instance, x: setattr(self.btn5, 'text', x))
        layout.add_widget(self.btn5)

        # Button widget

        # btn6 = Button(text='UPLOAD', font_size="20", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1),
        #               size_hint=(0.20, 0.10), pos_hint={'x': .1, 'y': 0.02})
        # btn6.bind(on_press=self.upload)
        # layout.add_widget(btn6)

        # Capture is for starting the video
        btn1 = Button(text='CAPTURE', font_size="20", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1),
                      size_hint=(0.20, 0.10), pos_hint={'x': .3, 'y': 0.02})
        btn1.bind(on_press=self.submit)
        layout.add_widget(btn1)

        # Snap for taking the picture
        btn2 = Button(text='SNAP', font_size="20", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1),
                      size_hint=(.20, .10), pos_hint={'x': .5, 'y': 0.02})
        btn2.bind(on_press=self.snapshot)
        layout.add_widget(btn2)

        # Reset for re-taken an image
        btn3 = Button(text='RESET', font_size="20", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1),
                      size_hint=(0.20, 0.10), pos_hint={'x': .7, 'y': 0.02})
        btn3.bind(on_press=self.reset)
        layout.add_widget(btn3)

        return layout

    # Reset function
    def reset(self, dt):
        self.capture = cv2.VideoCapture(0)
        self.submit(None)

    # Submit Function
    def submit(self, obj):
        trans_from = self.btn4.text
        print(trans_from)
        trans_to = self.btn5.text
        print(trans_to)

        LangFrom = trans_from
        LangTo = trans_to

        ArrayLang = ['afrikaans', 'amharic', 'arabic', 'azerbaijani', 'belarusian', 'bengali', 'bosnian', 'bulgarian',
                     'catalan', 'cebuano', 'chinese (simplified)', 'chinese (traditional)', 'croatian', 'czech',
                     'danish', 'dutch', 'english', 'esperanto', 'estonian', 'french', 'finnish', 'galician', 'georgian',
                     'german', 'hebrew', 'hindi', 'hungarian', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh',
                     'korean', 'khmer', 'kurdish', 'lao', 'latin', 'latvian', 'lithuanian', 'macedonian', 'malay',
                     'malayalam', 'maltese', 'marathi', 'myanmar', 'nepali', 'norwegian', 'pashto', 'polish',
                     'portuguese', 'romanian', 'russian', 'serbian', 'sinhala', 'slovak', 'slovenian', 'spanish',
                     'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'ukrainian', 'urdu', 'uzbek',
                     'vietnamese', 'yiddish']
        ArrayOCR = [
            "afr", "amh", "ara", "aze", "bel", "ben", "bos", "bul", "cat", "ceb", "chi_sim", "chi_tra", "hrv,", "ces",
            "dan", "deu", "eng", "epo", "est", "fra", "fin", "glg", "kat", "frk", "heb", "hin", "hun", "ita", "jpn",
            "jav", "kan", "kaz", "kor", "khm", "kur", "lao", "lat", "lav", "lit", "mkd", "msa", "mal", "mlt", "mar",
            "mya", "nep", "nor", "pus", "pol", "por", "ron", "rus", "srp", "sin", "slv", "slk", "spa", "swa", "swe",
            "tgk", "tam", "tel", "tha", "tur", "ukr", "urd", "uzb", "vie", "yid"]

        ArrayText = ['af', 'am', 'ar', 'az', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'zh-cn', 'zh-tw', 'hr', 'cs', 'da',
                     'nl', 'en', 'eo', 'et', 'fr', 'fi', 'gl', 'ka', 'de', 'he', 'hi', 'hu', 'it', 'ja', 'jw', 'kn',
                     'kk', 'ko', 'km', 'ku', 'lo', 'la', 'lv', 'lt', 'mk', 'ms', 'ml', 'mt', 'mr', 'my', 'ne', 'no',
                     'ps', 'pl', 'pt', 'ro', 'ru', 'sr', 'si', 'sk', 'sl', 'es', 'sw', 'sv', 'tg', 'ta', 'te', 'th',
                     'tr', 'uk', 'ur', 'uz', 'vi', 'yi']

        indexLangFrom = ArrayLang.index(LangFrom)
        indexLangTo = ArrayLang.index(LangTo)

        # Clock for keep taking frames from the video
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return ArrayOCR, indexLangFrom, ArrayText, indexLangTo, LangTo

    # def upload(self, file_path):
    #     fc = FileChooserIconView(title="Choose Image")
    #     image_path = fc.selection[0]
    #     image_name = file_path.split('/')[-1]
    #     buf1 = cv2.flip(image_name, 0)
    #     buf = buf1.tostring()
    #     texture1 = Texture.create(size=(image_name.shape[1], image_name.shape[0]), colorfmt='rgb')
    #     texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    #     self.img1.texture = texture1

    # Update function for the video display
    def update(self, dt):
        # Read the frame from the cv2.VideoCapture(0)
        ret, frame = self.capture.read()

        # Line display for helping the user taking the image
        cv2.line(img=frame, pt1=(325, 150), pt2=(325, 350), color=(255, 0, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(200, 250), pt2=(450, 250), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

        if ret:
            # Display the image on Kivy app
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture1

    # Snapshot function
    def snapshot(self, dt):
        # Read the frame at button press
        ret, frame = self.capture.read()
        # Stop the video
        self.capture.release()

        ArrayOCR, indexLangFrom, ArrayText, indexLangTo, LangTo = self.submit(self)

        if ret:
            # Select region of interest
            img_raw = frame.copy()
            ROIs = cv2.selectROIs("Select Region of interest", img_raw)
            print(ROIs)
            for rect in ROIs:
                x1 = rect[0]  # x
                y1 = rect[1]  # y
                x2 = rect[2]  # width
                y2 = rect[3]  # height

                # crop roi from original image
                img_crop = img_raw[y1:y1 + y2, x1:x1 + x2]
                cv2.destroyAllWindows()

                # Show cropped image after grayscale
                gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                cv2.imshow('gray', gray)

                # Pytesseract detect image
                options = "-l {} --psm {} --oem {}".format(ArrayOCR[indexLangFrom], args["psm"], args["oem"])
                text = pytesseract.image_to_string(gray, config=options)
                # show the original OCR'd text
                print("ORIGINAL")
                print("========")
                print(text)
                print("")

                # Google translator translate the OCR'd text
                translator = Translator()
                # Keep repeating if api is banned by Google
                while True:
                    try:
                        textTranslate = translator.translate(text, src=ArrayText[indexLangFrom],
                                                             dest=ArrayText[indexLangTo])
                        break
                    except Exception as e:
                        translator = Translator()

                print('Translation')
                print(f'source: {textTranslate.src}')
                print(f'Destination: {textTranslate.dest}')
                print(f'{textTranslate.text}')
                print("")

                # Display the image on Kivy at the same coordinates of the cropped image
                if LangTo == 'arabic':
                    reshaped_text = arabic_reshaper.reshape(textTranslate.text)
                    text = get_display(reshaped_text)
                    al = "right"
                    font_path = "arial.ttf"
                    font = ImageFont.truetype(font_path, 40)
                    img_pil = ImagePIL.fromarray(img_raw)
                    draw = ImageDraw.Draw(img_pil)
                    draw.rectangle((x1, y1, x1 + x2, y1 + y2), fill=(173, 171, 170))
                    draw.text((x1, y1), text, font=font, align=al, fill=(255, 0, 0))
                    img_raw = np.array(img_pil)
                else:
                    text = textTranslate.text
                    al = "left"
                    font_path = "arial.ttf"
                    font = ImageFont.truetype(font_path, 40)
                    img_pil = ImagePIL.fromarray(img_raw)
                    draw = ImageDraw.Draw(img_pil)
                    draw.rectangle((x1, y1, x1 + x2, y1 + y2), fill=(173, 171, 170))
                    draw.text((x1, y1), text, font=font, align=al, fill=(255, 0, 0))
                    img_raw = np.array(img_pil)

            buf2 = cv2.flip(img_raw, 0)
            buf = buf2.tostring()
            texture2 = Texture.create(size=(img_raw.shape[1], img_raw.shape[0]), colorfmt='bgr')
            texture2.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture2


if __name__ == '__main__':
    UCITA().run()
    cv2.destroyAllWindows()
