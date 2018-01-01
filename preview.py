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


def draw_line(tab, point1, point2, thick=False):
    K = 3
    if abs(point2[0] - point1[0]) > abs(point2[1] - point1[1]):
        a = (point1[1] - point2[1])/(point1[0] - point2[0])
        b = point1[1] - a * point1[0] 
        for x in range(point1[0], point2[0]):
            y = int(a * x + b) 
            tab[y, x , :] = 0xff
            if thick:
                for j in range(1, K):
                    tab[y+j, x , :] = 0xff
                    tab[y+j, x+j , :] = 0xff
                    tab[y, x+j , :] = 0xff
    else:
        a = (point1[0] - point2[0])/(point1[1] - point2[1])
        b = point1[0] - a * point1[1] 
        for x in range(point1[1], point2[1]):
            y = int(a * x + b) 
            tab[x, y , :] = 0xff
            if thick:
                for j in range(1, K):
                    tab[x+j, y , :] = 0xff
                    tab[x+j, y+j , :] = 0xff
                    tab[x, y+j , :] = 0xff


def draw_rec( contour):
    a = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)

    ### gora
    draw_line(a, (contour[0][0], contour[0][1]), (contour[1][0] , contour[1][1]), thick=True)
    ## dol
    draw_line(a, (contour[3][0], contour[3][1]), (contour[2][0] , contour[2][1]), thick=True)
    ## lewo
    draw_line(a, (contour[0][0], contour[0][1]), (contour[3][0] , contour[3][1]), thick=True)
    ## prawo
    draw_line(a, (contour[1][0], contour[1][1]), (contour[2][0] , contour[2][1]), thick=True)

    #print(contour[1][0], contour[3][0])

    ### RYSOWANIE GRIDU
    # up_diff_x = (contour[0][0] - contour[3][0]) // 9
    # up_diff_y = (contour[0][1] - contour[3][1]) // 9

    # side_diff_x = (contour[0][0] - contour[1][0]) // 9
    # side_diff_y = (contour[0][1] - contour[1][1]) // 9

    # print("roznice: ", up_diff_x, up_diff_y)

    # for x in range(0,10):
    #     thick = False
    #     if x % 3 == 0:
    #         thick = True
    #     draw_line(a, (contour[0][0] - x*up_diff_x, contour[0][1] - x*up_diff_y),
    #      (contour[1][0] - x*up_diff_x, contour[1][1] - x*up_diff_y), thick=thick)

    #     draw_line(a, (contour[0][0] - x*side_diff_x, contour[0][1] - x*side_diff_y),
    #      (contour[3][0] - x*side_diff_x, contour[3][1] - x*side_diff_y), thick=thick)

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

    o = camera.add_overlay(a.tobytes(), layer=3, alpha=15)
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


