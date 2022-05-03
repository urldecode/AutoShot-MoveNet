
from turtle import width
from cv2 import KeyPoint
from grpc import insecure_server_credentials
from requests import head 
import tensorflow as tf

import numpy as np
import pyautogui
import threading
import win32api
import win32con
import sys
import mss
import mss.tools
import time
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
tf.compat.v1.Session(config=config)

#定义屏幕尺寸和中心点
FullScrenn_WH = [1920,1080]
#Screen_Center = [FullScrenn_W/2,FullScrenn_H/2]
Screen_Center = [960,540]

#定义射击检测窗口大小
ScreenShot_WH = [512,512]

#定义射击框左上点坐标
#ScreenShot_Zero =[(1920-512)/2,(1080-512)/2]
ScreenShot_Zero =[704,284]
#定义射击框中心坐标
ScreenShot_Center = [256,256]


height = ScreenShot_WH[1]
width  = ScreenShot_WH[0]
keypoint_threshold = 0.20


# print("\n屏幕尺寸 : \t",FullScrenn_WH,
#     "\n屏幕中心点 : \t",Screen_Center,
#     "\n射击框左上角:\t",ScreenShot_Zero,
#     "\n射击框中心 : \t",ScreenShot_Center,
#     "\n射击框尺寸 : \t",ScreenShot_WH,
#     "\nheight大小 : \t",height,
#     "\nwidth 大小 : \t",width,)
# sys.exit()

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

        header_center = keypoints_all[0][3]
        header_center = header_center.astype(np.int16).tolist()
        Shot_X = header_center[0] - ScreenShot_Center[0] 
        Shot_Y = header_center[1] - ScreenShot_Center[1] 

        print("置信度：",'%.3f' % kpts_scores[3],"\t相对坐标：",header_center,"\t鼠标移动坐标：",Shot_X,Shot_Y,end="")
        #print(keypoints_all)
 
interpreter = tf.lite.Interpreter(model_path="./lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
interpreter.allocate_tensors()

def movenet(input_image):
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores



is_screen = False

class ScreenThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global is_screen
        while True:
            if is_screen is True:
                ScreenShot()

def ScreenShot():

    # image = pyautogui.screenshot(region=[ScreenShot_Zero[0],ScreenShot_Zero[1],ScreenShot_WH[0],ScreenShot_WH[1]])
    with mss.mss() as ts:
        monitor = {'top': 284 , 'left':704, 'width': 512, 'height': 512}
        sct_img = ts.grab(monitor)
        image = mss.tools.to_png(sct_img.rgb, sct_img.size)
        return image
#ttt=0
if __name__ =="__main__":
    while True:
        try:
            #time1=time.time()
            image = ScreenShot()
            #print("截图时间:\t",time.time()-time1)
            image = tf.image.decode_png(image)
            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, 192, 192)

            # Run model inference.
            time2=time.time()
            keypoints_with_scores = movenet(input_image)
            _keypoints_and_edges_for_display(keypoints_with_scores,height,width,keypoint_threshold)
            if Shot_Score > 0.4:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,Shot_X,Shot_Y,0,0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
            time3=time.time()-time2
            print("\t识别射击时间：\t", '%.3f' % time3)

        except:
            #print("未检测到 maybe error")
            pass
