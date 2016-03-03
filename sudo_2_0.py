# encoding: utf-8

import numpy as np
import xlrd
from copy import *
import sys


def encode(set_):
    list_ = list(set_)
    sorted(list_)
    result = 0
    for i in xrange(0, len(list_)):
        result += list_[i] * 10 ** i
    return result


def decode(number):
    list = []
    for value in str(number):
        list.append(int(value))
    return set(list)


class Matrix:
    def __init__(self, matrix_):
        '''初始化，matrix 为要填充的矩阵，matrix_left为每一格可能可以填的数'''
        self.matrix = copy(matrix_)
        self.left_matrix = np.matrix(np.zeros(9 * 9).reshape(9, 9), dtype = 'int')
        for i in xrange(0,9):
            for j in xrange(0,9):
                self.left_matrix[i,j] = encode(self.get_left(i,j))

        self.paichufa()

    def get_left(self, i, j):
        if self.matrix[i, j] < 0.1:
            return set(range(1, 10)) - set(self.matrix[i, :].reshape(1, 9).tolist()[0]) - set(self.matrix[:, j].reshape(1, 9).tolist()[0]) - set(self.matrix[3 * (i / 3):3 * (i / 3) + 3, 3 * (j / 3):3 * (j / 3) + 3].reshape(1, 9).tolist()[0])
        else:
            return set([])

    def update(self, value, i, j):
        #更新行：
        for k in xrange(0, 9):
            self.left_matrix[i, k] = encode(decode(self.left_matrix[i, k]) - set([value]))
        #更新列：
        for k in xrange(0, 9):
            self.left_matrix[k, j] = encode(decode(self.left_matrix[k, j]) - set([value]))

        #更新宫：
        ii = i/3
        jj = j/3
        for ki in xrange(3*ii, 3*ii+3):
            for kj in xrange(3*jj, 3*jj+3):
                self.left_matrix[ki, kj] = encode(decode(self.left_matrix[ki, kj]) - set([value]))

        self.left_matrix[i, j] = 0

    def paichufa(self):
        for k in xrange(0, 81):
            for i in xrange(0, 9):
                for j in xrange(0, 9):
                    self.solve_single(i, j)

    def illegal(self):
        for i in xrange(0, 9):
            for j in xrange(0, 9):
                if self.matrix[i, j] < 0.1 and self.get_left(i, j).__len__() == 0:
                    return True
        return False

    def finish(self):
        for i in xrange(0, 9):
            for j in xrange(0, 9):
                if self.matrix[i, j] < 0.1:
                    return False
        return True

    def solve_single(self, i, j):

        if self.left_matrix[i,j] == 0:
            return

        init = decode(self.left_matrix[i, j])

        if init.__len__() == 1:
            for value in init:
                self.matrix[i, j] = value
                self.update(value, i, j)
                return

        a = copy(init)
        for k in xrange(0, 9):
            if k != j:
                a = a - decode(self.left_matrix[i, k])
                if a == set([]):
                    break
        if a.__len__() == 1:
            for value in a:
                self.matrix[i, j] = value
                self.update(value, i, j)
                return

        b = copy(init)
        for k in xrange(0, 9):
            if k != i:
                b = b - decode(self.left_matrix[k, j])
                if b == set([]):
                    break
        if b.__len__() == 1:
            for value in b:
                self.matrix[i, j] = value
                self.update(value, i, j)
                return

        c = copy(init)
        for k1 in xrange(3 * (i / 3), 3 * (i / 3) + 3):
            for k2 in xrange(3 * (j / 3), 3 * (j / 3) + 3):
                if not (k1 == i and k2 == j):
                    c = c - decode(self.left_matrix[k1, k2])
                    if c == set([]):
                        break
        if c.__len__() == 1:
            for value in c:
                self.matrix[i, j] = value
                self.update(value, i, j)
                return

    def find_empty(self):
        for i in xrange(0, 9):
            for j in xrange(0, 9):
                if self.matrix[i, j] < 0.5:
                    return i, j
# 尾递归优化
class TailRecurseException:

    def __init__(self,args,kwargs):
        self.args = args
        self.kwargs = kwargs

def tail_call_optimized(g):

    def func(*args,**kwargs):
        f = sys._getframe()
        if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException, e:
                        args = e.args
                        kwargs = e.kwargs

    return func


class Solution_Tree_Node:

    def __init__(self, matrix_):
        self.data = Matrix(matrix_)
        self.child = []
        self.finish = self.data.finish()
        self.illegal = self.data.illegal()

class Solution_Tree:

    def __init__(self, root_matrix):
        self.root = Solution_Tree_Node(root_matrix)
        self.finish = False
        self.build(self.root)
        print self.complete_matrix


    # @tail_call_optimized
    def build(self, root_node):
        if self.finish:
            return
        else:
            if root_node.finish:
                self.finish = True
                self.complete_matrix = copy(root_node.data.matrix)
                # print self.complete_matrix
                return
            elif root_node.illegal:
                return
            else:
                i, j = root_node.data.find_empty()
                left = decode(root_node.data.left_matrix[i, j])
                root_matrix_copy = copy(root_node.data.matrix)
                for value in left:
                    root_matrix_copy[i, j] = value
                    new_matrix = root_matrix_copy
                    print new_matrix
                    new_node = Solution_Tree_Node(new_matrix)
                    new_node.parent = root_node
                    root_node.child.append(new_node)
                    self.build(new_node)


def importdata(filename):
    ''' 输入电子表格文件名，将电子表格中的数据输出为一个矩阵，一行为一个对象 '''
    data = xlrd.open_workbook(filename)
    datasheet = data.sheet_by_name("Sheet1")
    rows = datasheet.nrows
    cols = datasheet.ncols
    dataset = []
    for i in range(0, rows):
        dataset.append([])
        for j in range(0, cols):
            if datasheet.cell(i, j).value != '':
                dataset[i].append(int(datasheet.cell(i, j).value))
            else:
                dataset[i].append(0)
    return np.matrix(dataset)


m = importdata('data.xlsx')
print m
aa = Solution_Tree(m)

