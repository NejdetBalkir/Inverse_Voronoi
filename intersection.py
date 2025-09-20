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

# import numpy as np

# def line_intersection_vertical(x, m, b):
#     """
#     Intersection of vertical line x=const with line y=m*x+b.
#     Returns None if the other line is also vertical (m=None).
#     """
#     if m is None:  # both vertical -> no unique intersection
#         return None
#     x_intersect = x
#     y_intersect = m * x + b
#     return [x_intersect, y_intersect]


# def line_intersection_horizontal(y, m, b):
#     """
#     Intersection of horizontal line y=const with line y=m*x+b.
#     Returns None if the other line is also horizontal (m=0).
#     """
#     if m is None:  # vertical line case is fine
#         # then line is vertical x=c, no slope intercept -> can't compute here
#         return None
#     if abs(m) < 1e-12:  # both horizontal -> no unique intersection
#         return None
#     y_intersect = y
#     x_intersect = (y - b) / m
#     return [x_intersect, y_intersect]


