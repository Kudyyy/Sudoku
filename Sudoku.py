import numpy as np
import cv2
import sys
from copy import deepcopy
from matplotlib import pyplot as plt
from model import Model
from multiprocessing import Process, Lock
from pylab import *
import homography
from PIL import Image
from scipy import ndimage


class SudokuSolver:

    def __init__(self):
        self.N = 9
        self.lock = Lock()
        self.H = 1
        self.K = 3

    def load_model(self, path):
        self.model = Model()
        self.model.load(path)

    def __perform_homography(self, x):
        global H
        fp = array([array([p[1],p[0],1]) for p in x]).T
        tp = array([[0,0,1],[0,300,1],[300,300,1],[300,0,1]]).T
        # estimate the homography
        H = homography.H_from_points(tp,fp)


    def find_sudoku_contour(self, image):
        height, width = image.shape
        orig = cv2.resize(image, (width // self.K, height // self.K))
        kernel_val = 11 // self.K if (11 // self.K) % 2 == 1 else (11 // self.K) + 1
        blur = cv2.GaussianBlur(orig,(kernel_val, kernel_val),0)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_val, kernel_val))


        # perform a morphology based on the previously computed kernel
        close = cv2.morphologyEx(blur,cv2.MORPH_CLOSE,kernel1)
        div = np.float32(blur)/(close)
        res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))

        # perform an adaptive threshold and find the contours
        thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
        ind, contours,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


        # find the sudoku gameboard by looking for the largest square in image
        biggest = None
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > (10000 / (self.K ** 2)):
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > max_area and len(approx)==4:
                    biggest = approx
                    max_area = area        

        # calculate the center of the square
        try:
            M = cv2.moments(biggest)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except:
            return None, None

        # find the location of the four corners
        for a in range(0, 4):
            # calculate the difference between the center 
            # of the square and the current point
            dx = biggest[a][0][0] - cx
            dy = biggest[a][0][1] - cy

            if dx < 0 and dy < 0:
                topleft = (biggest[a][0][0] * self.K, biggest[a][0][1] * self.K)
            elif dx > 0 and dy < 0:
                topright = (biggest[a][0][0] * self.K, biggest[a][0][1] * self.K)
            elif dx > 0 and dy > 0:
                botright = (biggest[a][0][0] * self.K, biggest[a][0][1] * self.K)
            elif dx < 0 and dy > 0:
                botleft = (biggest[a][0][0] * self.K, biggest[a][0][1] * self.K)

        # the four corners from top left going clockwise
        try:
            corners = []
            corners.append(topleft)
            corners.append(topright)
            corners.append(botright)
            corners.append(botleft)
        except:
            return None, None

        self.__perform_homography(corners)
        res_size = np.float32([[0,0],[300,0],[300,300],[0,300]])
        M = cv2.getPerspectiveTransform(np.float32(corners),res_size)
        sudoku = cv2.warpPerspective(image,M,(300,300))
        sudoku = cv2.adaptiveThreshold(sudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,15,7)
        return corners, sudoku


    def get_sudoku_as_list(self, sudoku):
        
        def onMatrix(x, y, w, h):
            center = ((2*x+w)/2,(2*y+h)/2)
            cord = tuple((map(lambda x : int(x/30.5 + 0.4) - 1, center)))
            cord = tuple((map(lambda x : 0 if x < 0 else x , cord)))
            return tuple((map(lambda x : 8 if x > 8 else x , cord)))


        grid_mask = self.__draw_grid_mask(sudoku)
        without_grid = cv2.subtract(sudoku, grid_mask) 
            
        sudoku_matrix = [[ 0 for x in range(0,9)] for x in range(0,9)]

        ind, contours, hierarchy = cv2.findContours(without_grid,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)
        t_sum = 0

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if float(w)/h > 0.3 and float(w)/h < 1.2 and w*h > 50:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                i, j = onMatrix(x, y, w, h)
                symbol = sudoku[y:y+h, x:x+h]
                symbol = cv2.resize(symbol, (30, 30)) 
                symbol = cv2.bitwise_not(symbol)
                sudoku_matrix[j][i] = self.model.predict(symbol)

        return sudoku_matrix


    def __draw_grid_mask(self, image):
        grid_mask = np.zeros( (300,300),np.uint8)
        kernel = np.ones((6,1),np.uint8)
        vert = cv2.erode(image,kernel, iterations = 1)
        ind, contours, hierarchy = cv2.findContours(vert,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if float(w)/h < 0.1:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.fillPoly(grid_mask, pts =[box], color=255)

        kernel = np.ones((1,6),np.uint8)
        horiz = cv2.erode(image,kernel,iterations = 1)
        ind, contours, hierarchy = cv2.findContours(horiz,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if float(w)/h > 10:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.fillPoly(grid_mask, pts =[box], color=255)
        kernel = np.ones((2,2),np.uint8)
        return cv2.dilate(grid_mask,kernel)


    def solve_sudoku(self, sudoku):
        state = self.__read(sudoku)
        return self.__solve(state)


    def __read(self, field):
        """ Read field into state (replace 0 with set of possible values) """

        state = deepcopy(field)
        for i in range(self.N):
            for j in range(self.N):
                cell = state[i][j]
                if cell == 0:
                    state[i][j] = set(range(1,10))

        return state

    def __done(self, state):
        """ Are we done? """

        for row in state:
            for cell in row:
                if isinstance(cell, set):
                    return False
        return True


    def __propagate_step(self, state):
        """ Propagate one step """

        new_units = False

        for i in range(self.N):
            row = state[i]
            values = set([x for x in row if not isinstance(x, set)])
            for j in range(self.N):
                if isinstance(state[i][j], set):
                    state[i][j] -= values
                    if len(state[i][j]) == 1:
                        state[i][j] = state[i][j].pop()
                        new_units = True
                    elif len(state[i][j]) == 0:
                        return False, None

        for j in range(self.N):
            column = [state[x][j] for x in range(self.N)]
            values = set([x for x in column if not isinstance(x, set)])
            for i in range(self.N):
                if isinstance(state[i][j], set):
                    state[i][j] -= values
                    if len(state[i][j]) == 1:
                        state[i][j] = state[i][j].pop()
                        new_units = True
                    elif len(state[i][j]) == 0:
                        return False, None

        for x in range(3):
            for y in range(3):
                values = set()
                for i in range(3*x, 3*x+3):
                    for j in range(3*y, 3*y+3):
                        cell = state[i][j]
                        if not isinstance(cell, set):
                            values.add(cell)
                for i in range(3*x, 3*x+3):
                    for j in range(3*y, 3*y+3):
                        if isinstance(state[i][j], set):
                            state[i][j] -= values
                            if len(state[i][j]) == 1:
                                state[i][j] = state[i][j].pop()
                                new_units = True
                            elif len(state[i][j]) == 0:
                                return False, None

        return True, new_units

    def __propagate(self, state):
        """ Propagate until we reach a fixpoint """
        while True:
            solvable, new_unit = self.__propagate_step(state)
            if not solvable:
                return False
            if not new_unit:
                return True


    def __solve(self, state):
        """ Solve sudoku """

        solvable = self.__propagate(state)

        if not solvable:
            return None

        if self.__done(state):
            return state

        for i in range(self.N):
            for j in range(self.N):
                cell = state[i][j]
                if isinstance(cell, set):
                    for value in cell:
                        new_state = deepcopy(state)
                        new_state[i][j] = value
                        solved = self.__solve(new_state)
                        if solved is not None:
                            return solved
                    return None




