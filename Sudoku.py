
import numpy as np
import cv2
import sys
from copy import deepcopy
from matplotlib import pyplot as plt
from model import Model


class SudokuSolver:

    def __init__(self):
        self.N = 9

    def load_model(self, path):
        self.model = Model()
        self.model.load(path)


    def load_mask(self, path):
        self.mask = cv2.imread(path,0)
        ret, self.mask = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        #self.mask = cv2.bitwise_not(self.mask)


    def __sortV(self, vert):
        vert = sorted(vert, key=lambda v : v[1])
        if vert[0][0] > vert[1][0]:
            swaper = vert[0]
            vert[0] = vert[1]
            vert[1] = swaper
        if vert[3][0] > vert[2][0]:
            swaper = vert[2]
            vert[2] = vert[3]
            vert[3] = swaper
        return vert


    def find_sudoku_contour(self, image):
        blurred = cv2.GaussianBlur(image, (17, 17), 0)
        bin_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
         cv2.THRESH_BINARY_INV, 15, 7)
        with_mask = cv2.subtract(bin_image, self.mask)
        kernel = np.ones((3,3),np.uint8)
        noises_red = cv2.morphologyEx(with_mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.dilate(noises_red,kernel)
        ind, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        c_max = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c_max)
        box = cv2.boxPoints(rect)
        return np.int0(box)


    def get_sudoku_as_list(self, image, box=None):
        
        def onMatrix(x, y, w, h):
            center = ((2*x+w)/2,(2*y+h)/2)
            cord = tuple((map(lambda x : int(x/30.5 + 0.4) - 1, center)))
            return tuple((map(lambda x : 0 if x < 0 else x , cord)))

        if box is None:
            box = self.find_sudoku_contour(image)
            
        res_size = np.float32([[0,0],[300,0],[300,300],[0,300]])
        box = self.__sortV(box)
        M = cv2.getPerspectiveTransform(np.float32(box),res_size)
        sudoku = cv2.warpPerspective(image,M,(300,300))
        bin_image = cv2.adaptiveThreshold(sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
         cv2.THRESH_BINARY_INV, 21, 7)
        kernel = np.ones((2,2),np.uint8)
        noises_red = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
        grid_mask = self.__draw_grid_mask(noises_red)
        without_grid = cv2.subtract(noises_red, grid_mask) 
            
        sudoku_matrix = [[ 0 for x in range(0,9)] for x in range(0,9)]

        ind, contours, hierarchy = cv2.findContours(without_grid,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if float(w)/h > 0.3 and float(w)/h < 1.2 and w*h > 50:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                i, j = onMatrix(x, y, w, h)
                symbol = noises_red[y:y+h, x:x+h]
                symbol = cv2.resize(symbol, (30, 30)) 
                #cord = "("+str(x)+","+str(y)+").jpg"
                #path = "wyniki/"
                symbol = cv2.bitwise_not(symbol)
                #cv2.imwrite(path+cord,symbol)
                result = self.model.predict(symbol)
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
            if float(w)/h > 5:
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




