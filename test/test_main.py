import pytest
import cv2 as cv
import numpy as np
import os
from src.main import region_of_interest, reflect_line_vertically, make_coordinates, display_polygon, perspective_transform

@pytest.fixture
def real_frame():
    frame_path = os.path.join("test_data", "redrock1_test_img.png")
    frame = cv.imread(frame_path)
    if frame is None:
        pytest.skip("File not found")

    # cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    # cv.resizeWindow('Frame', 1280, 720)
    # cv.imshow("Frame", frame)
    # cv.waitKey(0)

    return frame

def test_region_of_interest_real_frame(real_frame):
    roi = region_of_interest(real_frame)
    assert roi.shape == real_frame.shape
    assert np.any(roi > 0)

def test_reflect_line_vertically(real_frame):
    line = np.array([495, 1080, 1005,  648])
    width = real_frame.shape[1]
    reflected = reflect_line_vertically(line, real_frame)
    assert reflected[1] == line[1]
    assert reflected[3] == line[3]
    assert reflected[0] == width - line[0]
    assert reflected[2] == width - line[2]

def test_make_coordinates(real_frame):
    slope, intercept = -0.847, 1500
    coords = make_coordinates(real_frame, (slope, intercept))
    assert coords.shape == (4,)
    assert coords[1] == real_frame.shape[0]

def test_display_polygon(real_frame):
    left_line = np.array([500, 1080, 700, 600])
    right_line = np.array([1400, 1080, 1200, 600])
    lines = np.array([left_line, right_line])
    img, polygon = display_polygon(real_frame, lines, show_result=True)
    print(polygon.shape)
    assert img.shape == real_frame.shape
    assert polygon.shape == (1, 4, 2)

def test_perspective_transform(real_frame):
    polygon_points = np.array([[(600, 400), (1200, 400), (1600, 1080), (300, 1080)]], dtype=np.int32)
    transformed = perspective_transform(real_frame, polygon_points)
    assert transformed.shape == (800, 600, 3)
