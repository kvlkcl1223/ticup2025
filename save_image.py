import cv2
import os
import numpy as np
def undistort(frame):

    k=np.array([[1.96295802e+03, 0.00000000e+00, 9.04350359e+02],
                [0.00000000e+00, 1.95866974e+03, 5.68555114e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


    d=np.array([-0.36102308, -0.19379845, -0.00559319,  0.00637392,  1.47648705])
    h,w=frame.shape[:2]
    mapx,mapy=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),5)
    return cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)






camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
camera.set(cv2.CAP_PROP_FOURCC, fourcc)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1960)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# while True:
#     ret, frame = camera.read()
#     if ret:
#         frame = undistort(frame)
#         frame = cv2.rotate(frame, cv2.ROTATE_180)
#         frame = frame[400:1080, 620:1240]
#         cv2.imshow('frame', frame)
#         cv2.waitKey(1)

i = 0
ret, img = camera.read()
print('输入j,下载当前图片')
print('输入q,终止程序')
os.makedirs('./kuangtu', exist_ok=True)
while True:
    ret, img = camera.read()
    if ret:
        img = undistort(img)

        img=cv2.rotate(img, cv2.ROTATE_180)
        img = img[400:1080, 620:1240]
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('j'):  # 按j保存一张图片
            i += 1
            firename = str('./kuangtu/img' + str(i) + '.jpg')
            cv2.imwrite(firename, img)
            print('写入：', firename)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.release()
cv2.destroyAllWindows()
#