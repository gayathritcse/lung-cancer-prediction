from tkinter import *
import os
from tkinter import filedialog
import cv2

from tkinter import messagebox


def fulltraining():
    import EfficientNetB3 as mm


def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 500
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.configure()
    main_screen.title("Lung Disease Prediction ")

    Label(text="Lung Disease Prediction ", width="300", height="5", font=("Palatino Linotype", 16)).pack()


    Label(text="").pack()
    Button(text="Train", font=(
        'Palatino Linotype', 15), height="2", width="20", command=fulltraining, highlightcolor="black").pack(side=TOP)

    Label(text="").pack()

    Label(text="").pack()

    main_screen.mainloop()


main_account_screen()
