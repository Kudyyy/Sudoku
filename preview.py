import io
import sys
import cv2
import numpy
import time
import imutils
import math

from picamera import PiCamera
from picamera.array import PiRGBArray
from sudoku import SudokuSolver
from PIL import Image
from time import sleep
from scipy import ndimage

COLOR = 0x4F


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
    K = 5
    if abs(point2[0] - point1[0]) > abs(point2[1] - point1[1]):
        a = (point1[1] - point2[1])/(point1[0] - point2[0])
        b = point1[1] - a * point1[0] 
        for x in range(point1[0], point2[0]):
            y = int(a * x + b) 
            tab[y, x , :] = COLOR
            if thick:
                for j in range(1, K):
                    tab[y+j, x , :] = COLOR
                    tab[y+j, x+j , :] = COLOR
                    tab[y, x+j , :] = COLOR
    else:
        a = (point1[0] - point2[0])/(point1[1] - point2[1])
        b = point1[0] - a * point1[1] 
        for x in range(point1[1], point2[1]):
            y = int(a * x + b) 
            tab[x, y , :] = COLOR
            if thick:
                for j in range(1, K):
                    tab[x+j, y , :] = COLOR
                    tab[x+j, y+j , :] = COLOR
                    tab[x, y+j , :] = COLOR


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


def draw_solution_on_screen(sudoku, contour):
    text_layer = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = [' '.join(map(str, row)) for row in sudoku]
    
    y_diff = (contour[1][0]-contour[0][0])//9
    font_size = y_diff / 35

    rotation = - 57.3 * math.atan((contour[0][1]-contour[1][1])/(contour[0][0]-contour[1][0]))
    offset = y_diff
    top_left = list(contour[0])
    top_left[0] += y_diff//4
    top_right = list(contour[1])
    top_right[1] = top_left[1]
    down_right = [top_right[0], top_right[1] + 9 * y_diff]
    down_left = [top_left[0], top_left[1] + 9 * y_diff]
    for row in rows:
        cv2.putText(text_layer, row, (top_left[0], top_left[1] + offset), font, font_size, (255,255,255), 2, cv2.LINE_AA)
        offset += y_diff
        
    pts1 = numpy.float32([top_left, top_right,down_right, down_left])
    pts2 = numpy.float32(contour)

   # M = cv2.getAffineTransform(pts1,pts2)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    #text_layer = cv2.warpAffine(text_layer,M,(1280,720))
    text_layer = cv2.warpPerspective(text_layer,M,(1280,720))
    #text_layer = imutils.rotate(text_layer, rotation)
    return camera.add_overlay(text_layer.tobytes(), layer=3, alpha = 64)

    
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

    # print("Find sudoku contour: ", time)

    if contour is None:
        continue

    try:
        sudoku = solver.get_sudoku_as_list(sudoku)
        if sudoku is not None:
            a = draw_rec(contour)
            o = camera.add_overlay(a.tobytes(), layer=3, alpha = 84)
            drawn = True
            sudoku = solver.solve_sudoku(sudoku)
            if sudoku is None:
                camera.remove_overlay(o)
                raise Exception("Could not solve sudoku")
            print_field(sudoku)
            txt_overlay = draw_solution_on_screen(sudoku, contour)
            sleep(1)
            camera.remove_overlay(o)
            camera.remove_overlay(txt_overlay)
    except Exception as ex:
        print(ex)
        continue

