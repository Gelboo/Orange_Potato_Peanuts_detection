from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
from theClassify import classify
from thedetect import detection
from PIL import Image, ImageTk

filename = ""
root = Tk()

root.bind('<Escape>', lambda e: root.quit())
root.attributes('-fullscreen', True)
root.configure(background='blue')

img_panel = Label(root)
img_panel.place(x=100,y=100,width=800,height=800)

classifyLbl = Label(root,text="Type: ",bg = 'blue',font='Calibri 22')
classifyLbl.place(x=1200,y=200)

classifyRes = Label(root,text='NULL',fg='yellow',bg='blue',font='Calibri 22')
classifyRes.place(x=1300,y=200)

detectLbl = Label(root,text="Status: ",bg ='blue',font='Calibri 22')
detectLbl.place(x=1200,y=300)

detectRes = Label(root,text='NULL',bg='blue',fg='yellow',font='Calibri 22')
detectRes.place(x=1330,y=300)

def load_image():
    global img_panel,filename,detectRes,classifyRes
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()
    img = ImageTk.PhotoImage(file=filename,size=(10,10))
    img_panel.config(image=img)
    img_panel.image = img

    classifyRes.config(text="NULL")
    classifyRes.text="NULL"
    classifyRes.config(fg="yellow")
    classifyRes.fg="yellow"

    detectRes.config(text="NULL")
    detectRes.text="NULL"
    detectRes.config(fg="yellow")
    detectRes.fg="yellow"

def classify_img():
    global filename,classifyRes
    img = cv2.resize(cv2.imread(filename),(100,100))
    lbl = classify(img)
    classifyRes.config(text=lbl)
    classifyRes.text=lbl
    classifyRes.config(fg="white")
    classifyRes.fg="white"

def detect_img():
    global filename,detectRes
    img = cv2.resize(cv2.imread(filename),(100,100))
    status = detection(img)
    detectRes.config(text=status)
    detectRes.text=status
    if status == "Approved":
        detectRes.config(fg="white")
        detectRes.fg="white"
    elif status == "defected":
        detectRes.config(fg="red")
        detectRes.fg="red"

load = Button(root,text="loadImage",command=load_image,width=8,height=2,bg='black',fg='white',font='Calibri 20')
load.place(x=400,y=950)

classifyButton = Button(root,text="classify",command=classify_img,width=8,height=2,bg='black',fg='white',font='Calibri 20')
classifyButton.place(x=650,y=950)

detectButton = Button(root,text="detect",command=detect_img,width=8,height=2,bg='black',fg='white',font='Calibri 20')
detectButton.place(x=900,y=950)



root.mainloop()
# image,label,status = load_image()
# print(label+" "+status)
# cv2.imshow("image",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
