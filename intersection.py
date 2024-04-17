import numpy as np

def line_intersection_vertical(x,m,b):

    x_intersect = x
    y_intersect = m*x+b

    intersection_point = [x_intersect,y_intersect]

    return intersection_point

def line_intersection_horizontal(y,m,b):
    y_intersect = y
    x_intersect = (y-b)/m

    intersection_point = [x_intersect,y_intersect]

    return intersection_point