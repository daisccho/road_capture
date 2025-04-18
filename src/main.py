import cv2 as cv
import numpy as np
import argparse
from tqdm import tqdm


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines, prev_left_line, prev_right_line):
    left_fit = []
    right_fit = []

    if lines is None:
        if prev_left_line is not None:
            left_line = prev_left_line
        else:
            left_line = None
        if prev_right_line is not None:
            right_line = prev_right_line
        else:
            right_line = None
    else:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(image, left_fit_average)
            prev_left_line = left_line
        else:
            left_line = prev_left_line

        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(image, right_fit_average)
            prev_right_line = right_line
        else:
            right_line = prev_right_line

        if left_line is None:
            left_line = reflect_line_vertically(right_line, image)
            prev_left_line = left_line
        if right_line is None:
            right_line = reflect_line_vertically(left_line, image)
            prev_right_line = right_line

    return np.array([left_line, right_line]), np.array([prev_left_line, prev_right_line])


def reflect_line_vertically(line, image):
    width = image.shape[1]
    x1, y1, x2, y2 = line
    x1_reflected = width - x1
    x2_reflected = width - x2
    return np.array([x1_reflected, y1, x2_reflected, y2])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 10)
    return line_image


def display_polygon(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        left_line, right_line = lines
        x1_left, y1_left, x2_left, y2_left = left_line
        x1_right, y1_right, x2_right, y2_right = right_line
        height = image.shape[0]
        y_upper = height // 1.3
        y_lower = 1080

        x_left_upper = int(x1_left + (y_upper - y1_left) * (x2_left - x1_left) / (y2_left - y1_left))
        x_left_lower = int(x1_left + (y_lower - y1_left) * (x2_left - x1_left) / (y2_left - y1_left))
        x_right_upper = int(x1_right + (y_upper - y1_right) * (x2_right - x1_right) / (y2_right - y1_right))
        x_right_lower = int(x1_right + (y_lower - y1_right) * (x2_right - x1_right) / (y2_right - y1_right))

        polygon_points = np.array([[(x_left_upper, y_upper), (x_right_upper, y_upper), (x_right_lower, y_lower), (x_left_lower, y_lower)]], np.int32)
        cv.polylines(line_image, polygon_points, isClosed=True, color=(255, 255, 255), thickness=10)

    return line_image, polygon_points


def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([(700, height // 1.3), (1120, height // 1.3), (1370, 1080), (395, 1080)])
    polygon_negative = np.array([(880, height // 1.3), (970, height // 1.3), (1170, 1080), (640, 1080)])
    mask = np.zeros_like(image)
    mask_neg = np.zeros_like(image)
    cv.fillPoly(mask, np.array([polygon], dtype=np.int64), color=(255, 255, 255))
    cv.fillPoly(mask_neg, np.array([polygon_negative], dtype=np.int64), color=(255, 255, 255))
    res = cv.subtract(mask, mask_neg)
    roi = cv.bitwise_and(image, res)
    return roi


def adapt_region_of_interest(image, previous_lines):
    frame_copy = cv.GaussianBlur(image, (5, 5), 1)
    hls = cv.cvtColor(frame_copy, cv.COLOR_RGB2HLS)
    lower_white = np.array([0, 150, 0])
    upper_white = np.array([180, 255, 80])
    white_mask = cv.inRange(hls, lower_white, upper_white)
    lower_yellow = np.array([80, 0, 65])
    upper_yellow = np.array([105, 180, 255])
    yellow_mask = cv.inRange(hls, lower_yellow, upper_yellow)
    mask = cv.bitwise_or(white_mask, yellow_mask)
    result = cv.bitwise_and(frame_copy, frame_copy, mask=mask)
    result = cv.cvtColor(result, cv.COLOR_RGB2GRAY)
    result = cv.equalizeHist(result)

    adaptive_thresh = cv.adaptiveThreshold(result, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 20)
    adaptive_thresh = region_of_interest(adaptive_thresh)

    lines = cv.HoughLinesP(adaptive_thresh, 1, np.pi / 270, 90, np.array([()]), minLineLength=50, maxLineGap=5)
    averaged_lines, previous_lines = average_slope_intercept(adaptive_thresh, lines, previous_lines[0], previous_lines[1])
    line_image, polygon_points = display_polygon(frame_copy, averaged_lines)
    result = cv.addWeighted(frame_copy, 0.8, line_image, 0.5, 1)
    return result, previous_lines, polygon_points


def perspective_transform(image, polygon_points, target_points=np.array([[0, 0], [600, 0], [600, 800], [0, 800]], dtype=np.float32)):
    matrix = cv.getPerspectiveTransform(polygon_points.astype(np.float32), target_points)
    transformed_image = cv.warpPerspective(image, matrix, (600, 800))
    return transformed_image


def process_video(video_path, output_path):
    video = cv.VideoCapture(video_path)
    frames_total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    cur_frame = 1
    delay = 1
    delta = 20

    if not video.isOpened():
        print("Couldn't fetch the video")
        return

    previous_lines = np.array([None, None])
    result_image = None

    with tqdm(total=frames_total) as pbar:
        while cur_frame <= frames_total:
            ret, frame = video.read()
            if not ret:
                break

            frame_copy = np.copy(frame)
            cur_frame += 1

            try:
                result, previous_lines, polygon_points = adapt_region_of_interest(frame_copy, previous_lines)
                warped = perspective_transform(frame, polygon_points)

                if result_image is None:
                    black_block = np.zeros((delta * (frames_total - 1), warped.shape[1], 3), dtype=np.uint8)
                    result_image = np.copy(warped)
                    result_image = np.vstack((black_block, result_image))
                elif cur_frame < frames_total:
                    y_offset = (warped.shape[0] + delta * (frames_total - 1)) - (warped.shape[0] + cur_frame * delta)
                    result_image[y_offset:y_offset + warped.shape[0], :warped.shape[1]] = warped

                pbar.update(1)

            except Exception as e:
                print('Exception: ', e)
                pass

            if cv.waitKey(delay) == 27:
                break

    cv.imwrite(output_path, result_image)

    video.release()
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Process a video and save the result")
    parser.add_argument("\'\\path\\to\\input_video.mp4\'", help="Path to the input video file")
    parser.add_argument("\'\\path\\to\\output_image\'", help="Path to save the output image")
    args = parser.parse_args()

    process_video(args.input_video, args.output_image)


if __name__ == "__main__":
    main()
