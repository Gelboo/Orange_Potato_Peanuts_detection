from tkinter import *
import time
import cv2
from imutils.video import VideoStream
from PIL import Image, ImageTk
import threading
import imutils
from thresholding import getContours

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
root.attributes('-fullscreen', True)
root.configure(background='black')

frame = None
thread = None
stopEvent = None
panel = None
vs = VideoStream(0).start()

def exit():
        global root,vs,stopEvent
        vs.stop()
        stopEvent.set()
        stop(0.3)

def videoLoop():
        global frame,stopEvent,panel,vs,root
        try:
            # keep looping over frames until we are instructed to stop
            while not stopEvent.is_set():
                    # grab the frame from the video stream and resize it to
                    # have a maximum width of 300 pixels
                    frame = vs.read()

                    # frame = cv2.imread('Pictures/Potato/Potato_16.jpg')
                    # frame = cv2.videoCapture(0)

                    frame = imutils.resize(frame, width=600,height=600)

                    # OpenCV represents images in BGR order; however PIL
                    # represents images in RGB order, so we need to swap
                    # the channels, then convert to PIL and ImageTk format


                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)


                    # if the panel is not None, we need to initialize it
                    if panel is None:
                            txt = Label(text="Original",bg='black',fg='blue',font='Calibri 22')
                            txt.place(x=350,y=20)
                            panel = Label(image=image)
                            panel.image = image
                            panel.place(x = 100 ,y=80)

                    # otherwise, simply update the panel
                    else:
                            panel.configure(image=image)
                            panel.image = image

        except:
                print("[INFO] caught a RuntimeError")

panel2 = None
panel3 = None
panel4 = None
cnt = 0
def videoLoop2():
        global frame,stopEvent,panel2,vs,root,cnt
        try:
            # keep looping over frames until we are instructed to stop
            while not stopEvent.is_set():
                    # grab the frame from the video stream and resize it to
                    # have a maximum width of 300 pixels
                    frame = vs.read()


                    frame = imutils.resize(frame, width=600,height=600)

                    # OpenCV represents images in BGR order; however PIL
                    # represents images in RGB order, so we need to swap
                    # the channels, then convert to PIL and ImageTk format


                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    img = Image.fromarray(img)
                    img = ImageTk.PhotoImage(img)



                    # if the panel is not None, we need to initialize it
                    if panel2 is None:
                            txt = Label(text="Result",bg='black',fg='blue',font='Calibri 22')
                            txt.place(x=1200,y=20)
                            panel2 = Label(image=img)
                            panel2.image = img
                            panel2.place(x=950,y=80)
                    # otherwise, simply update the panel
                    elif cnt%100 == 0:
                            panel2.configure(image=img)
                            panel2.image = img
                            print("inter",cnt)
                    cnt += 1
                    print("exter",cnt)

        except:
                print("[INFO] caught a RuntimeError")




def onClose():
        global stopEvent,vs,root
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        stopEvent.set()
        vs.stop()
        root.quit()

stopEvent = threading.Event()
thread = threading.Thread(target=videoLoop, args=())
thread2 = threading.Thread(target=videoLoop2, args=())
thread.start()
thread2.start()

# set a callback to handle when the window is closed
root.wm_title("PyImageSearch PhotoBooth")
root.wm_protocol("WM_DELETE_WINDOW", onClose)

root.mainloop()
