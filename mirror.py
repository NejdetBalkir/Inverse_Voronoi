import numpy as np
import math

def mirror_point(point, line_point1, line_point2):
    x, y = point
    x1, y1 = line_point1
    x2, y2 = line_point2



    # Calculate the slope of the line
    m = (y2 - y1) / (x2 - x1)

    # assuming that the equation of line is y = mx + c
    # We can find c as follows:
    c = y1 - m*x1

    # Calculate the perpendicular distance from the point to the line
    distance = abs(m*x - y + c) / (m**2 + 1)**(0.5)

    # unit vector between two points
    u = line_point2 - line_point1
    unit_u = u/((x2-x1)**2 + (y2-y1)**2)**(0.5)
    unit_normal = [-unit_u[1],unit_u[0]]

    v = np.array([x-x1,y-y1])

    dot_product = np.dot(unit_normal,v)


    if (dot_product < 0):
        unit_normal = -np.array(unit_normal)

    

    # Calculate the coordinates of the mirrored point
    mirrored_x = x - 2 * distance * unit_normal[0]
    mirrored_y = y - 2 * distance * unit_normal[1]

    mirror_point = [mirrored_x,mirrored_y]

    return mirror_point