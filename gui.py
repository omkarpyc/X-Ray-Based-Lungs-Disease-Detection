import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

from tensorflow.keras.models import load_model

model = load_model('model_res152V2new.h5')
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# dictionary to label all traffic signs class.

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Lungs Disease Detection using Resnet 152V2 ')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    test_image = load_img(file_path, target_size=(224, 224))  # load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # change dimention 3D to 4D

    result = model.predict(test_image).round(3)  # predict diseased palnt or not
    print('@@ Raw result = ', result)

    pred = np.argmax(result)  # get the index of max value

    if pred == 0:
        print("Bacterial Pneumonia")
        label.configure(foreground='#011638', text="Diseased pair of lungs, Name of disease: Bacterial Pneumonia")
    elif pred == 1:
        print('Corona Virus Disease'),
        label.configure(foreground='#011638', text="Diseased pair of lungs, Name of disease: Corona Virus Disease")
    elif pred == 2:
        print('Normal'),
        label.configure(foreground='#011638', text="Healthy pair of lungs")
    elif pred == 3:
        print('Tuberculosis'),
        label.configure(foreground='#011638', text="Diseased pair of lungs, Name of disease: Tuberculosis")
    elif pred == 4:
        print('Viral Pneumonia'),
        label.configure(foreground='#011638', text="Diseased pair of lungs, Name of disease: Viral Pneumonia")

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path),
                        padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25),
                            (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Lungs Disease Detection", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
