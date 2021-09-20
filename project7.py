from sympy import *
from itertools import *
import numpy as np
from colorama import Fore, Back, Style


def read_input():
    """
    --- Description
    Read input

    --- Return
    :return: - m = number of rows (type: <int>)
             - n = number of columns (type: <int>)
    """
    m = int(input("m= "))
    n = int(input("n= "))
    return m, n


def get_all_matrices(n, k):
    """
    Get all combinations of matrices with elements 0 and 1
    :param n: number of rows (type: <int>)
    :param k: number of columns (type: <int>)
    :return: list of lists with all matrices
    """
    return [np.reshape(np.array(i), (n, k)) for i in product([0, 1], repeat=k*n)]


def rref(B, tol=1e-8):
    """
    --- Description
    Implement the reduced row echelon form algorithm.
    Reduced row echelon form has four requirements:
        - 1. the first non-zero number in the first row (the leading entry) is the number 1.
        - 2. the second row also starts with the number 1, which is further to the right than the leading entry in the
          first row. For every subsequent row, the number 1 must be further to the right.
        - 3. the leading entry in each row must be the only non-zero number in its column.
        - 4. any non-zero rows are placed at the bottom of the matrix.

    --- Parameters
    :param B: the matrix (type: <numpy matrix>)
    :param tol: tolerance for 0
    :return: the rref matrix and the leading positions
    """
    A = B.copy()  # make a copy
    rows, cols = A.shape  # get the nr of rows and cols
    r = 0
    leading_pos = []  # list with the leading 1's
    row_exchanges = np.arange(rows)  # =[0, rows)

    for c in range(cols):
        # Find the pivot row:
        pivot = np.argmax(np.abs(A[r:rows, c])) + r  # the index of the |maximum value| in the column
        max_val = np.abs(A[pivot, c])  # the |maximum value|
        if max_val <= tol:
            # if the maximum value is zero, then all the column is formed by 0's. We don't care and move on
            A[r:rows, c] = np.zeros(rows - r)
        else:
            # the maximum value isn't zero. We save the maximum value as a leading position
            leading_pos.append((r, c))

            if pivot != r:
                # If the pivot isn't the current row, then just swap the rows. (to respect second condition)
                A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
                row_exchanges[[pivot, r]] = row_exchanges[[r, pivot]]

            # Normalize pivot row in Z2
            A[r, c:cols] = (A[r, c:cols] / A[r, c]) % 2

            # Eliminate the current column
            v = A[r, c:cols]
            # Subtract from the above row, the actual row:
            if r > 0:
                rid_x_above = np.arange(r)
                A[rid_x_above, c:cols] = (A[rid_x_above, c:cols] - np.outer(v, A[rid_x_above, c]).T) % 2

            # Subtract from the below row, the actual row:
            if r < rows - 1:
                rid_x_below = np.arange(r + 1, rows)
                A[rid_x_below, c:cols] = (A[rid_x_below, c:cols] - np.outer(v, A[rid_x_below, c]).T) % 2

            # go to the next row
            r += 1
        # Check if done
        if r == rows:
            break
    return A, leading_pos


def start_algorithm(possibilities):
    """
    --- Description
    Start the algorithm

    --- Parameters
    :param possibilities: the list of lists with all the matrix possibilities

    --- Return
    :return: the number of rref matrices and the list of them
    """
    final_list = []
    leading_pos_final = []
    nr = 0
    for matrix in possibilities:
        M, leading_pos = rref(matrix)  # get the row reduced echelon form
        ok = 1
        for j in range(len(final_list)):  # check if the rref already exists in the final list
            if np.all(M == final_list[j]):
                ok = 0
        if ok:  # if it doesn't exist in the list, add it to the list and increase the number
            nr += 1
            final_list.append(M)
            leading_pos_final.append(leading_pos)
    return nr, final_list, leading_pos_final


def print_number_matrices(nr):
    print("The number of matrices A belonging to M(m,n)(Z2) in reduced echelon form is " + str(nr), file=g)


def print_number_matrices_terminal(nr):
    print("The number of matrices A belonging to M(m,n)(Z2) in reduced echelon form is " + str(nr))


def print_matrices(lst, leading_list, m, n):
    """
    Print matrices in reduced echelon form are (the leading 1’s are highlighted):

    :param lst: the list of all rref matrices
    :param leading_list: the list of all leading positions
    :param m: the nr of rows
    :param n: the nr of columns
    """
    for k in range(len(lst)):
        matrix = lst[k]
        leading_pos = leading_list[k]
        index_pos = 0
        for i in range(m):
            for j in range(n):
                if len(leading_pos):
                    if i == leading_pos[index_pos][0] and j == leading_pos[index_pos][1]:
                        print(str(matrix[i][j]), end=" ", file=g)
                        if index_pos+1 < len(leading_pos):
                            index_pos += 1
                    else:
                        print(matrix[i][j], end=" ", file=g)
                else:
                    print(matrix[i][j], end=" ", file=g)
            print(end="\n", file=g)
        print(end="\n", file=g)


def print_matrices_terminal(lst, leading_list, m, n):
    """
    Print matrices in reduced echelon form are (the leading 1’s are highlighted):

    :param lst: the list of all rref matrices
    :param leading_list: the list of all leading positions
    :param m: the nr of rows
    :param n: the nr of columns
    """
    for k in range(len(lst)):
        matrix = lst[k]
        leading_pos = leading_list[k]
        index_pos = 0
        for i in range(m):
            for j in range(n):
                if len(leading_pos):
                    if i == leading_pos[index_pos][0] and j == leading_pos[index_pos][1]:
                        print(Back.CYAN + Fore.BLACK + str(matrix[i][j]) + Style.RESET_ALL, end=" ")
                        if index_pos+1 < len(leading_pos):
                            index_pos += 1
                    else:
                        print(matrix[i][j], end=" ")
                else:
                    print(matrix[i][j], end=" ")
            print(end="\n")
        print(end="\n")


if __name__ == '__main__':
    with open('project7/test1.in', 'r') as f:
        m, n = [int(x) for x in next(f).split()]
    # choose from:
    # test1.in, test2.in, test3.in, test4.in, test5.in

    # IF YOU WANT THE LEADING POSITIONS TO BE HIGHLIGHTED PLEASE SET text_highlight = True. I can't frame/color
    # text in output file, only in terminal. I searched and I didn't find anything helpful/that worked.
    text_highlight = False

    if not text_highlight:
        g = open('project7/test1.out', 'w+')
    # choose from:
    # test1.out, test2.out, test3.out, test4.out, test5.out

    list_of_possibilities = get_all_matrices(m, n)
    nr, rref_list, leading_list = start_algorithm(list_of_possibilities)
    if not text_highlight:
        print_number_matrices(nr)
    else:
        print_number_matrices_terminal(nr)
    if 2 <= m <= 5 and 2 <= n <= 5:
        if not text_highlight:
            print_matrices(rref_list, leading_list, m, n)
        else:
            print_matrices_terminal(rref_list, leading_list, m, n)
