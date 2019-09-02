#imports from AoisDefiner
import tkinter as tk
from tkinter import Label,Tk
from PIL import Image, ImageTk
import PIL.Image
import json
from tkinter import messagebox
from tkinter import filedialog
from tkinter import *
import csv

coords =[]
class App(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self._createVariables(parent)
        self._createCanvas()
        self._createCanvasBinding()

    def _createVariables(self, parent):
        self.parent = parent
        self.rectx0 = 0
        self.recty0 = 0
        # self.rectx1 = 0
        # self.recty1 = 0
        # self.rectid = None
        self.move = False

    def _createCanvas(self):
        self.canvas = tk.Canvas(self.parent, width=1633, height=768,
                                bg="white")
        self.canvas.grid(row=0, column=0, sticky='nsew')
        path = filedialog.askopenfilename(filetypes=[("Image File", '.jpg')])
        self.im = PIL.Image.open(path)
        self.wazil, self.lard = self.im.size
        self.canvas.config(scrollregion=(0, 0, self.wazil, self.lard))
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)

    def _createCanvasBinding(self):
        self.canvas.bind("<Button-1>", self.startRect)

    def startRect(self, event):
        self.move = True
        # Translate mouse screen x0,y0 coordinates to canvas coordinates
        self.rectx0 = self.canvas.canvasx(event.x)
        self.recty0 = self.canvas.canvasy(event.y)
        # Create rectangle
        self.rect = self.canvas.create_rectangle(
            self.rectx0, self.recty0, self.rectx0, self.recty0)
        # Get rectangle's canvas object ID
        self.rectid = self.canvas.find_closest(self.rectx0, self.recty0, halo=2)
        print('Rectangle {0} started at {1} {2} {3} {4} '.
              format(self.rect, self.rectx0, self.recty0, self.rectx0,
                     self.recty0))
        window = tk.Tk()

        window.title("Area of Interest")
        labelone = tk.Label(window, text="Name of Area of Interest")
        labelone.grid(row=0, column=0)
        name = tk.StringVar()

        userEntry = tk.Entry(window, textvariable=name)
        userEntry.grid(row=0, column=1)


        def addAOI():
            string_answer = userEntry.get()
            item = {
                'fixation_number': string_answer, 'x': self.rectx0, 'y': self.recty0,
            }
            coords.append(item)
            print(coords)
            close_window()

        def close_window():
            window.destroy()

        def fnProgram():
            window.destroy()
            main()

        def saveJson():
            with open('puntos.json', 'w') as outfile:
                json.dump(coords, outfile)
                alert = tk.Tk()

                alert.title("Success")
                label_a = tk.Label(alert, text="The output was saved")
                label_a.grid(row=0, column=0)
                button_a = tk.Button(alert, text="Ok", command=fnProgram)
                button_a.grid(row=1, column=0)

        btn = tk.Button(window, text="Add", command=addAOI)
        btn.grid(row=0, column=3)
        btn = tk.Button(window, text="Finish AOI definition", command=saveJson)
        btn.grid(row=1, column=1)

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1366x768")
    root.attributes("-fullscreen", True)
    app = App(root)
    root.mainloop()