### 検出したマーカーのIDをリストで返す
import cv2
from cv2 import aruco
import numpy as np
import time


class MarkSearch:
    ### --- aruco設定 --- ###
    dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    def __init__(self, cameraID):
        self.cap = cv2.VideoCapture(cameraID)

    def get_markID(self):
        """
        静止画を取得し、arucoマークのidリストを取得する
        """
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        list_ids = np.ravel(ids)

        return list_ids


if __name__ == "__main__":
    import cv2
    from cv2 import aruco
    import numpy as np
    import time

    ### --- aruco設定 --- ###
    dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    ### --- parameter --- ###
    cameraID = 1  #0:default laptop's camera, 1:webcamera
    cam0_mark_search = MarkSearch(cameraID)

    try:
        while True:
            print(' ----- get_markID ----- ')
            print(cam0_mark_search.get_markID())
            time.sleep(0.5)
    except KeyboardInterrupt:
        cam0_mark_search.cap.release()