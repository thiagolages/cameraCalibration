# https://docs.opencv.org/4.6.0/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2 as cv
import glob
import numpy as np

width  = 6
height = 8

calibration_path = 'calibrationImages/*.png'

print("Starting calibration on folder: "+calibration_path)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(width,5,0)
objp = np.zeros((width*height,3), np.float32)
objp[:,:2] = np.mgrid[0:height,0:width].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
imgnames = []
images = glob.glob(calibration_path)
for idx, fname in enumerate(images):
    print("Loading image: "+str(idx), end="\r", flush=True)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (height,width), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        imgnames.append(fname)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (height,width), corners2, ret)
        cv.imshow('img', img)
        #cv.waitKey(500)
        if (cv.waitKey(1) == ord('q')):
            cv.destroyAllWindows()
            exit()
cv.destroyAllWindows()

print("Calibrating parameters, please wait..")

# Calibration
ret, cameraMatrix, distortionCoeff, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret, cameraMatrix, distortionCoeff, rvecs, tvecs: ",ret, cameraMatrix, distortionCoeff, rvecs, tvecs)


# save

np.savetxt("cameraMatrix.txt"       , cameraMatrix      , delimiter=",")
np.savetxt("cameraDistortion.txt"   , distortionCoeff   , delimiter=",")

# f = open("cameraMatrix.txt", "w")
# cameraMatrix_np = np.array(cameraMatrix)
# for row in range(cameraMatrix_np.shape[0]):
#     for col in range(cameraMatrix_np.shape[1]):
#         f.write(str(cameraMatrix_np[row,col]))
#         if (col < cameraMatrix_np.shape[1] - 1):
#             f.write(",")
    
#     if (row < cameraMatrix_np.shape[0] - 1):
#         f.write("\n")

# f.close()



# f = open("cameraDistortion.txt", "w")
# distortionCoeff_np = np.array(distortionCoeff)
# for row in range(distortionCoeff_np.shape[0]):
#     for col in range(distortionCoeff_np.shape[1]):
#         f.write(str(distortionCoeff_np[row,col]))
#         if (col < distortionCoeff_np.shape[1] - 1):
#             f.write(",")
    
#     if (row < distortionCoeff_np.shape[0] - 1):
#         f.write("\n")

# f.close()


# Undistortion

img = cv.imread('calibrationImages/CU81/CU81_image19.png')


# Re-projection error

mean_error = 0
for i in range(len(objpoints)):
    
    img = cv.imread(imgnames[i])
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distortionCoeff)
    ret = None
    cv.drawChessboardCorners(img, (height,width), imgpoints2, ret)
    cv.imshow('distorted image with reprojection', img)


    h,  w = img.shape[:2]
    # Alpha - scaling parameter
    # If alpha=0, it returns undistorted image with minimum unwanted pixels. So it may even remove some pixels at image corners. 
    # If alpha=1, all pixels are retained with some extra black images.
    alpha = 0
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distortionCoeff, (w,h), alpha, (w,h))

    # simple undistort
    dst = cv.undistort(img, cameraMatrix, distortionCoeff, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    simple_undistort = dst[y:y+h, x:x+w]
    #cv.imwrite('calibresult.png', dst)
    cv.imshow('simple undistort', simple_undistort)

    # complex undistort (remapping)
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distortionCoeff, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    complex_undistort = dst[y:y+h, x:x+w]
    #cv.imwrite('calibresult.png', dst)
    cv.imshow('complex undistort', complex_undistort)

    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

    cv.waitKey(0)


print( "total error: {}".format(mean_error/len(objpoints)) )

cv.waitKey(0)