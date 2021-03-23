import numpy as np
import cv2
import time

def calc_unfish_map(img_in, k, d, dims):
    dim1 = img_in.shape[:2][::-1]

    # print(dim1)
    assert dim1[0] / dim1[1] == dims[0] / dims[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    
    nk = k.copy()
    nk[0,0]=k[0,0]/2
    nk[1,1]=k[1,1]/2
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), nk, dims, cv2.CV_16SC2)
    
    # print(map1.shape)
    # print(map2.shape)
    
    # print(map1)
    # print(map2)


    # img_out = cv2.remap(img_in, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # return img_out

    map1,map2 = cv2.convertMaps(	map1, map2, cv2.CV_16SC2)


    return map1,map2



def calc_tan_map(src,h_angle):    
    w = src.shape[1]
    h = src.shape[0]
    v_angle = h_angle/w*h
    s = (h,w,2)

    map1 = np.zeros(s,dtype=np.float32)
    # cv2.CV_16SC2
    # map1 = cv2.createmat(h,w,cv2.CV_32FC2)
    map2= None   

    x_curve = np.linspace(-h_angle/2,h_angle/2,w)

    #ratio = TAN(Radians(angle))/TAN(RADIANS(60))
    x_curve = np.tan(np.radians(x_curve))/np.tan(np.radians(h_angle/2))

    x_curve = (x_curve+1)/2*(w-1)
    
    y_curve = np.linspace(-v_angle/2,v_angle/2,h)

    #ratio = TAN(Radians(angle))/TAN(RADIANS(60))
    y_curve = np.tan(np.radians(y_curve))/np.tan(np.radians(v_angle/2))

    y_curve = (y_curve+1)/2*(h-1)

    map1[::,::,0] = x_curve
    # map1[::,::,1] = np.repeat(np.transpose([np.linspace(0,h-1,h)]),repeats=w,axis=1)
    map1[::,::,1] = np.repeat(np.transpose([y_curve]),repeats=w,axis=1)

    map1,map2 = cv2.convertMaps(	map1, map2, cv2.CV_16SC2)

    return map1,map2

    
    # start = time.time() 
    # dst = cv2.remap(src, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)        
    # end = time.time() 
    # print("% s seconds" % (end-start)) 

    # return dst



if __name__ == '__main__':
    Dims = tuple(np.load('./parameters/Dims.npy'))
    K = np.load('./parameters/K.npy')
    D = np.load('./parameters/D.npy')
    
    img = cv2.imread('distort.jpg')

    map1,map2=calc_unfish_map(img, k=K, d=D, dims=Dims)

    start = time.time() 
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)  
    end = time.time() 
    print("% s seconds" % (end-start)) 

    cv2.imwrite('undistorted.jpg', img)
    
    map1,map2 = calc_tan_map(img,125)

    start = time.time() 
    img_tan = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)   
    end = time.time() 
    print("% s seconds" % (end-start)) 
    
    
    cv2.imshow('', img)     
    cv2.waitKey(2000)
    cv2.imshow('', img_tan) 
    cv2.waitKey(2000)

    cv2.imwrite('undistorted.jpg', img)
    cv2.imwrite('linear.jpg', img_tan)
    
    cv2.destroyAllWindows()
    
    
    # img = correct(img, k=K, d=D, dims=(640,480))
