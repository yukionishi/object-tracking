import cv2
import numpy as np

# Initialize ARUCO detector
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Pixel-to-mm conversion factor
width, height = (500, 500)
real_distance_cm = 200 #cm
pixel_distance = 500  # in img_trans
conversion_factor = real_distance_cm / pixel_distance

# Access the webcam
cap = cv2.VideoCapture(1)
address = "https://10.115.128.205:8080/video"
cap.open(address)
contrast = 0
brightness = -50
# ret, frame = cap.read()
# corners, ids, _ = aruco.detectMarkers(frame, dictionary)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # adjusted_img = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # Detect AR markers
    # corners, ids, _ = aruco.detectMarkers(adjusted_img, dictionary)
    corners, ids, _ = aruco.detectMarkers(frame, dictionary)

    # If at least five markers are detected (4 corners + 1 additional)
    if ids is not None and len(ids) >= 5:
        # Separate the corners and the additional marker
        m = np.zeros((4,2))
        additional_marker_corners = None

        transCorners = [np.empty((1, 4, 2))] * 4

        for i, c in zip(ids.ravel(), corners):
            # If the id is between 0 and 3, it's a corner marker
            if i in [0, 1, 2, 3]:
                transCorners[i] = c.copy()
                m[0] = transCorners[0][0][2]  # id=0(left top), position of 2(right bottom)
                m[1] = transCorners[1][0][3]  # id=1(right top), position of 3(left bottom)
                m[2] = transCorners[2][0][0]  # id=2(right bottom), position of 0(left top)
                m[3] = transCorners[3][0][1]  # id=3(left bottom), position of 1(right top)
            else:
                additional_marker_corners = c[0]

        # Get transformation matrix
        marker_coordinates = np.float32(m)
        true_coordinates   = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        trans_matrix = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)
        img_trans = cv2.warpPerspective(frame, trans_matrix, (width, height))

        # Get the transformed position of the additional marker's center
        additional_marker_center = additional_marker_corners.mean(axis=0).reshape(1,1,2)
        additional_marker_transformed = cv2.perspectiveTransform(additional_marker_center, trans_matrix)

        real_position_x = additional_marker_transformed[0][0][0] * conversion_factor
        real_position_y = additional_marker_transformed[0][0][1] * conversion_factor
        real_position = [real_position_x, real_position_y]
        print("Real Position of Additional Marker (cm):", real_position)

        # Save or display the result
        cv2.imshow("Transformed Image", img_trans)
        # cv2.imwrite('output.jpg', img_trans)  # If you want to save the image

    # Show original webcam feed with detected markers
    aruco.drawDetectedMarkers(frame, corners, ids)
    # cv2.imshow("Webcam Feed", adjusted_img)
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()