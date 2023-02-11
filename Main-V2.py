#这个文件为最新, 使用方法 python Main-V2.py
# 优化了截图方式,性能提升

from turtle import right, width
import cv2
from grpc import insecure_server_credentials
from requests import head 
import tensorflow as tf
import numpy as np
import pyautogui
#import threading
import win32api
import win32gui
import win32ui
import win32con

import mss.tools
import time
import numpy

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.Session(config=config)

#避免枪口上抬，下压像素8值
drop= 5
#两发子弹间隔时间-
    #awp                 0.13        1           
    #AK47       0.1      0.08        5-10        
    #m16A4      0.075    < 0.07      5           16
    #m16A4-2    0.075    0           5-15
    #沙鹰        0.222
#shottime = 0.02
shottime = 0.05

# 检测值，置信度 
keypoint_threshold = 0.3

#射击置信度最低值
Shot_Score_2 = 0.32

#定义屏幕尺寸和中心点
FullScrenn_WH = [1920,1080]
#定义射击检测窗口大小
ScreenShot_WH = [192,192]

#定义射击框左上点坐标
Zero_X = int((FullScrenn_WH[0]-ScreenShot_WH[0])/2)
Zero_Y = int((FullScrenn_WH[1]-ScreenShot_WH[1])/2)
ScreenShot_Zero =[Zero_X,Zero_Y]
#ScreenShot_Zero =[832,412]
#定义射击框中心坐标
ScreenShot_Center = [int(ScreenShot_WH[0]/2),int(ScreenShot_WH[1]/2)]

height = ScreenShot_WH[1]
width  = ScreenShot_WH[0]

def _keypoints_and_edges_for_display(keypoints_with_scores,height,width,keypoint_threshold):
    keypoints_all = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        #nose = keypoints_all[0][1]
        # left_ear  = keypoints_all[0][3]
        # right_ear = keypoints_all[0][4]
        # #left_should  = keypoints_all[0][5]
        # #right_should = keypoints_all[0][6]
        # header_center = (left_ear + right_ear) /2
        global Shot_X 
        global Shot_Y
        global Shot_Score
        
        Shot_Score = kpts_scores[3]

        left_should = keypoints_all[0][3]
        right_should = keypoints_all[0][4]

        

        header_center = (left_should + right_should) /2

        header_center = header_center.astype(np.int16).tolist()
        Shot_X = header_center[0] - ScreenShot_Center[0] 
        Shot_Y = header_center[1] - ScreenShot_Center[1] 
        

interpreter = tf.lite.Interpreter(model_path="./lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
interpreter.allocate_tensors()

def movenet(input_image):
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


def ScreenShot():

    hdesktop = win32gui.GetDesktopWindow()
    desktop_dc = win32gui.GetWindowDC(hdesktop)
    img_dc = win32ui.CreateDCFromHandle(desktop_dc)
    mem_dc = img_dc.CreateCompatibleDC()
    screenshot = win32ui.CreateBitmap()
    screenshot.CreateCompatibleBitmap(img_dc, width, height)
    mem_dc.SelectObject(screenshot)
    mem_dc.BitBlt((0, 0), (width, height), img_dc, (Zero_X, Zero_Y), win32con.SRCCOPY)

    bmpstr = screenshot.GetBitmapBits(True)
    im_opencv = numpy.frombuffer(bmpstr, dtype = 'uint8')
    im_opencv.shape = (height, width, 4)
    image = cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2RGB)

    img_dc.DeleteDC()
    mem_dc.DeleteDC()
    
    win32gui.DeleteObject(screenshot.GetHandle())
    win32gui.ReleaseDC(hdesktop,desktop_dc)

    return image

if __name__ =="__main__":
    while True:
        try:
            if win32api.GetKeyState(win32con.VK_RBUTTON) < 0: # 右键
                time1=time.time()

                image = ScreenShot()

                input_image = tf.expand_dims(image, axis=0)
                input_image = tf.image.resize_with_pad(input_image, 192, 192)

                keypoints_with_scores = movenet(input_image)
                _keypoints_and_edges_for_display(keypoints_with_scores,height,width,keypoint_threshold)

                if (Shot_Score > Shot_Score_2) :
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,Shot_X,Shot_Y+drop,0,0)

                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

                    time.sleep(shottime)   
                    time2 = time.time()
                    print("检测到射击时间:\t", time2 - time1)
                    # 截图+检测大约30ms
    
        except:
            pass
          
