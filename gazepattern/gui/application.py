# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django

#python
import tkinter as tk
from tkinter import Label,Tk
from PIL import Image, ImageTk
import PIL.Image
import json
#gazepattern
from eyedetector.models import Image, ImageRectangle


class Application(object):

    def __init__(self, image_path, image):
        root = tk.Tk()
        root.geometry( "1000x720" )
        self.aois = []
        self.parent = root
        self.path = image_path
        self.image = image
        self.move = False
        self._create_canvas()
        self._create_canvas_binding()
        root.mainloop()

    def _create_canvas(self):
        self.canvas = tk.Canvas(self.parent, width = 1366, height = 768,
                                bg = "white" )
        self.canvas.grid(row=0, column=0, sticky='nsew')
        #path=filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
        self.im = PIL.Image.open(self.path)
        self.wazil,self.lard=self.im.size
        self.canvas.config(scrollregion=(0,0,self.wazil,self.lard))
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)

    def _create_canvas_binding(self):
        self.canvas.bind( "<Button-1>", self.start_rect )
        self.canvas.bind( "<ButtonRelease-1>", self.stop_rect )
        self.canvas.bind( "<Motion>", self.moving_rect )

    def start_rect(self, event):
        self.move = True
        #Translate mouse screen x0,y0 coordinates to canvas coordinates
        self.rectx0 = self.canvas.canvasx(event.x)
        self.recty0 = self.canvas.canvasy(event.y) 
        #Create rectangle
        self.rect = self.canvas.create_rectangle(
            self.rectx0, self.recty0, self.rectx0, self.recty0)
        #Get rectangle's canvas object ID
        self.rectid = self.canvas.find_closest(self.rectx0, self.recty0, halo=2)
        print('Rectangle {0} started at {1} {2} {3} {4} '.
              format(self.rect, self.rectx0, self.recty0, self.rectx0,
                     self.recty0))

    def moving_rect(self, event):
        if self.move: 
            #Translate mouse screen x1,y1 coordinates to canvas coordinates
            self.rectx1 = self.canvas.canvasx(event.x)
            self.recty1 = self.canvas.canvasy(event.y)
            #Modify rectangle x1, y1 coordinates
            self.canvas.coords(self.rectid, self.rectx0, self.recty0,
                          self.rectx1, self.recty1)
            print('Rectangle x1, y1 = ', self.rectx1, self.recty1)

    def stop_rect(self, event):
        """
        Una vez que se termina de seleccionar el área, se muestra la pantalla de tkinter dónde se debe ingresar el nombre del área
        
        """


        self.move = False
        #Translate mouse screen x1,y1 coordinates to canvas coordinates
        self.rectx1 = self.canvas.canvasx(event.x)
        self.recty1 = self.canvas.canvasy(event.y) 
        #Modify rectangle x1, y1 coordinates (final)
        self.canvas.coords(self.rectid, self.rectx0, self.recty0,
                      self.rectx1, self.recty1)
        window = tk.Tk()

        window.title("Area of Interest")
        labelone = tk.Label(window, text="Name of Area of Interest")
        labelone.grid(row = 0, column = 0)
        name = tk.StringVar()

        userEntry = tk.Entry(window, textvariable = name)
        userEntry.grid(row = 0, column = 1)

        def addAOI():
            """
            Se agrega al JSON el elemento con el nombre y sus coordenadas
            """

            string_answer = userEntry.get()
            item = {
            'x0': self.rectx0, 'y0': self.recty0,
            'x1': self.rectx1, 'y1': self.recty1,
            'aoi': string_answer
            }

            self.aois.append(item)
            print(self.aois)
            close_window()
        
        def close_window(): 
            window.destroy()

        def finish_program():
            self.parent.destroy()

        def save_json():
            with open('data.json', 'w') as outfile:
                
                self.image.rectangles.all().delete()
                for aoi in self.aois:
                    rectangle = ImageRectangle()
                    rectangle.x0 = aoi.get('x0')
                    rectangle.x1 = aoi.get('x1')
                    rectangle.y0 = aoi.get('y0')
                    rectangle.y1 = aoi.get('y1')
                    rectangle.name = aoi.get('aoi')
                    rectangle.image = self.image
                    rectangle.save()
                    print(rectangle)

                close_window()
                finish_program()

        btn = tk.Button(window, text ="Add", command = addAOI)
        btn.grid(row = 0, column = 3)
        btn = tk.Button(window, text ="Finish AOI definition", command = save_json)
        btn.grid(row = 1, column = 1)

