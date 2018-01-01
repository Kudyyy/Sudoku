from Sudoku import SudokuSolver
import cv2
import sys

def output(a):
    sys.stdout.write(str(a))

def print_field(field):
    if not field:
        output("No solution")
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
       
e1 = cv2.getTickCount()
solver = SudokuSolver()
solver.load_model("./model.h5")
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("init",time)

image = cv2.imread('img0.jpg',0)
image = cv2.resize(image, (960, 540)) 
print(type(image))
#plt.figure()
#plt.imshow(image)

e1 = cv2.getTickCount()
contour = solver.find_sudoku_contour(image)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("find_contour",time)

e1 = cv2.getTickCount()
sudoku = solver.get_sudoku_as_list(image)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("get_sudo_as_list",time)
print(sudoku)

e1 = cv2.getTickCount()
sudoku = solver.solve_sudoku(sudoku)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("solve_sud",time)
print_field(sudoku)

e1 = cv2.getTickCount()
contour = solver.find_sudoku_contour(image)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("find_contour",time)

e1 = cv2.getTickCount()
sudoku = solver.get_sudoku_as_list(image)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("get_sudo_as_list",time)
print(sudoku)

e1 = cv2.getTickCount()
sudoku = solver.solve_sudoku(sudoku)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("solve_sud",time)
print_field(sudoku)
