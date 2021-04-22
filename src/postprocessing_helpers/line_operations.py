import cv2
import numpy as np


def find_vertices_of_prediction_joints(prediction_edges, minimum_joint_prediction_height=100):
    """
    Finds top and bottom vertices of prediction joints.

    :param minimum_joint_prediction_height: Minimum height of predicted joint to be considered as valid prediction
    :param prediction_edges: output of canny edge operation on prediction mask
    :return: list of top and bottom vertices for each predicted joint.
    """
    contours, hierarchy = cv2.findContours(prediction_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vertices_list = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Skip detected joints that are not high enough
        if h < minimum_joint_prediction_height:
            continue

        pts = contour.reshape(contour.shape[0], 2)
        top_left, bottom_left, top_right, bottom_right = find_corners_in_points(pts)

        # Find top and bottom vertice as averate of top right/left and bottom right/left
        x_top = int((top_left[0] + top_right[0]) / 2)
        y_top = int((top_left[1] + top_right[1]) / 2)

        x_bot = int((bottom_left[0] + bottom_right[0]) / 2)
        y_bot = int((bottom_left[1] + bottom_right[1]) / 2)

        vertices_list.append((x_top, y_top, x_bot, y_bot))

    return vertices_list


def find_corners_in_points(pts: np.ndarray):
    s = np.sum(pts, axis=1)
    top_left = pts[np.argmin(s)]  # Top left corner has minimum sum
    bottom_right = pts[np.argmax(s)]  # Bottom right corner has maximum sum

    diff = np.diff(pts, axis=1)
    top_right = pts[np.argmin(diff)]  # Top right has minimum difference
    bottom_left = pts[np.argmax(diff)]  # Bottom left has maximum difference

    return top_left, bottom_left, top_right, bottom_right


def get_middle_point_of_lines(vertices_list):
    """
    Finds middle point of line

    :param vertices_list: list of vertices. Vertices should be in following order: x_top, y_top, x_bot, y_bot
    :return: list of dictionaries containing coordinates of middle point and vertices of corresponding line
    """

    line_middle_points = []
    for index, vertices in enumerate(vertices_list):
        xm, ym = get_middle_point(vertices[0], vertices[1], vertices[2], vertices[3])
        line_middle = {"middle_point": (xm, ym), "vertices": vertices}
        line_middle_points.append(line_middle)
    return line_middle_points


def get_general_line_coeffs(xa, ya, xb, yb):
    """
    Calculates general line coefficients based on line top and bottom vertices

    :param xa:
    :param ya:
    :param xb:
    :param yb:
    :return: dictionary with general coefficients and corresponding vertices
    """
    A = ya - yb
    B = xb - xa
    C = ((-1) * ya * xb) + (xa * yb)
    coeffs = {"general_coeffs": (A, B, C), "vertices": (xa, ya, xb, yb)}
    return coeffs


def get_point_line_distance(coeffs, point):
    """
    Calculates distance between point and line

    :param coeffs: General coefficients of line
    :param point: Point coordinates
    :return:
    """
    A = coeffs[0]
    B = coeffs[1]
    C = coeffs[2]
    x0 = point[0]
    y0 = point[1]

    distance_nominator = np.abs((A * x0 + B * y0 + C))
    distance_denominator = np.sqrt(A ** 2 + B ** 2)
    distance = distance_nominator / distance_denominator
    return distance


def get_closest_lines_indexes(pred_middle_points, grayscale_lines_coeffs_vertices):
    """
    For each middle prediciton point finds closest line found on grayscale image

    :param pred_middle_points: list of dictionaries containing coordinates of middle point and vertices of
     corresponding line
    :param grayscale_lines_coeffs_vertices: dictionary with general coefficients and corresponding vertices
     for lines found on grayscale image
    :return:
    """
    closest_lines_indexes = []
    for index, point in enumerate(pred_middle_points):
        middle_point = point["middle_point"]
        distances = []
        for coeffs_vertices in grayscale_lines_coeffs_vertices:
            line_coeffs = coeffs_vertices["general_coeffs"]
            distance = get_point_line_distance(line_coeffs, middle_point)
            distances.append(distance)

        closest_lines_indexes.append(np.argmin(distances))

    return closest_lines_indexes


def get_middle_point(x1, y1, x2, y2):
    xm = int((x1 + x2) / 2)
    ym = int((y1 + y2) / 2)

    return xm, ym


def point_to_point_distance(x1, y1, x2, y2):
    distance = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    return distance


def get_point_on_opposite_edge_of_handler(front_bbox: np.ndarray, handler_bbox):

    handler_middle_pt_x = int(handler_bbox.x_offset + (handler_bbox.width / 2))
    handler_middle_pt_y = int(handler_bbox.y_offset + (handler_bbox.height / 2))

    # TODO: Try to get middle points of edges from masks not bboxes
    front_left_middle_point_x = front_bbox.x_offset
    front_left_middle_point_y = int(front_bbox.y_offset + (front_bbox.height / 2))

    front_right_middle_point_x = front_bbox.x_offset + front_bbox.width
    front_right_middle_point_y = int(front_bbox.y_offset + (front_bbox.height / 2))

    # Calculate the distance from handler to left edge and from handler to right edge
    left_distance = point_to_point_distance(front_left_middle_point_x, front_left_middle_point_y,
                                            handler_middle_pt_x, handler_middle_pt_y)
    right_distance = point_to_point_distance(front_right_middle_point_x,
                                             front_right_middle_point_y, handler_middle_pt_x, handler_middle_pt_y)

    # Based on the distance choose which front edge contains joint and return middle point of that edge
    if right_distance > left_distance:
        return front_right_middle_point_x, front_right_middle_point_y
    else:
        return front_left_middle_point_x, front_left_middle_point_y
