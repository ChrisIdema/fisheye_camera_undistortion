import numpy as np
import cv2


def correct(img_in, k, d, dims):
    dim1 = img_in.shape[:2][::-1]
    assert dim1[0] / dim1[1] == dims[0] / dims[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    
    nk = k.copy()
    nk[0,0]=k[0,0]/2
    nk[1,1]=k[1,1]/2
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), nk, dims, cv2.CV_16SC2)

    img_out = cv2.remap(img_in, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img_out


if __name__ == '__main__':
    Dims = tuple(np.load('./parameters/Dims.npy'))
    K = np.load('./parameters/K.npy')
    D = np.load('./parameters/D.npy')
    
    img = cv2.imread('distort.jpg')
    img = correct(img, k=K, d=D, dims=(640,480))
    cv2.imshow('', img)
    cv2.imwrite('undistorted.jpg', img)
