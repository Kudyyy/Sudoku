from picamera import PiCamera
from picamera.array import PiRGBArray
from Sudoku import SudokuSolver
from PIL import Image
import io
import sys
import cv2
import numpy
import time

def output(a):
    sys.stdout.write(str(a))

def print_field(field):
    if not field:
        output("no Solution")
        return
    N = 9
    for i in range(N):
        for j in range(N):
            cell = field[i][j]
            if cell == 0 or isinstance(cell, set):
                output('.')
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(' |')                                                    
            if j != 8:
                output(' ')
        output('\n')
        if (i + 1) % 3 == 0 and i < 8:
            output("- - - + - - - + - - -\n")

def draw_rec( contour):
    a = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)

    diff = (contour[0][1] - contour[1][1])/(contour[0][0] - contour[1][0])
    b = contour[0][1] - diff * contour[0][0]
    for x in range(contour[0][0], contour[1][0]):
        y = int(diff * x + b)
        a[y, x, :] = 0xff

    diff = (contour[3][1] - contour[2][1])/(contour[3][0] - contour[2][0])
    b = contour[3][1] - diff * contour[3][0]
    for x in range(contour[3][0], contour[2][0]):
        y = int(diff * x + b) 
        a[y, x, :] = 0xff
    

    diff = (contour[0][0] - contour[3][0])/(contour[0][1] - contour[3][1])
    b = contour[0][0] - diff * contour[0][1]
    for x in range(contour[0][1], contour[3][1]):
        y = int(diff * x + b) 
        a[x, y, :] = 0xff

    diff = (contour[1][0] - contour[2][0])/(contour[1][1] - contour[2][1])
    b = contour[1][0] - diff * contour[1][1]
    for x in range(contour[1][1], contour[2][1]):
        y = int(diff * x + b) 
        a[x, y, :] = 0xff

    return a 
    
stream = io.BytesIO()
solver = SudokuSolver()
solver.load_model("./model.h5")
camera = PiCamera()
camera.resolution = (1280, 720)
rawCapture = PiRGBArray(camera, size=(1280, 720))
camera.start_preview()
time.sleep(1)

drawn = False

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    rawCapture.truncate(0)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    e1 = cv2.getTickCount()
    contour, sudoku = solver.find_sudoku_contour(gray)
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print("Find sudoku contour: ", time)
    if contour is None:
        continue

    if drawn:
        drawn = False
        camera.remove_overlay(o)
    try:
        a = draw_rec(contour)
    except:
        continue

    o = camera.add_overlay(a.tobytes(), layer=3, alpha=64)
    drawn = True
        
    e1 = cv2.getTickCount()
    sudoku = solver.get_sudoku_as_list(sudoku)
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print("Get sudoku as list: ", time)
    print_field(sudoku)
    e1 = cv2.getTickCount()
    
    if sudoku is None:
        continue

    sudoku = solver.solve_sudoku(sudoku)
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print_field(sudoku)     
    print("Rest: ", time)   


