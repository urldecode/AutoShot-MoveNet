
import cv2
import tensorflow as tf
import numpy as np

import win32api
import win32gui
import win32ui
import win32con
import time
import numpy
import random
import ctypes

user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
keypoint_threshold = 0.31
FullScrenn_WH = [1920,1080]
ScreenShot_WH = [120,120]
Zero_X = int((FullScrenn_WH[0]-ScreenShot_WH[0])/2)
Zero_Y = int((FullScrenn_WH[1]-ScreenShot_WH[1])/2)
ScreenShot_Zero =[Zero_X,Zero_Y]
ScreenShot_Center = [int(ScreenShot_WH[0]/2),int(ScreenShot_WH[1]/2)]

height = ScreenShot_WH[1]
width  = ScreenShot_WH[0]

interpreter = tf.lite.Interpreter(model_path="./lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
interpreter.allocate_tensors()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.Session(config=config)

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("mi", MOUSEINPUT)]

def move_mouse(x, y):
    mi = MOUSEINPUT(x, y, 0, MOUSEEVENTF_MOVE, 0, None)
    inp = INPUT(ctypes.c_ulong(0), mi)
    user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

def click_mouse(x, y):
    mi = MOUSEINPUT(x, y, 0, MOUSEEVENTF_LEFTDOWN, 0, None)
    inp = INPUT(ctypes.c_ulong(0), mi)
    user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))
    mi = MOUSEINPUT(x, y, 0, MOUSEEVENTF_LEFTUP, 0, None)
    inp = INPUT(ctypes.c_ulong(0), mi)
    user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

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

def ScanAndShot():
    time1=time.time()
    image = ScreenShot()
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 192, 192)
    keypoints_with_scores = movenet(input_image)
    _keypoints_and_edges_for_display(keypoints_with_scores,height,width,keypoint_threshold)
    
    if (Shot_Score > keypoint_threshold) :
        drop =  random.randint(4,10) 
        move_mouse(Shot_X,Shot_Y)
        click_mouse(Shot_X,Shot_Y)
        shottime=round(random.uniform(0.05,0.08),3)
        time.sleep(shottime)   
        time2 = time.time()
        print("检测到射击时间:\t", int((time2 - time1)*1000), "ms\t 射击 间隔: \t", int(shottime*1000),"ms")
 
if __name__ =="__main__":
    while True:
        try:
            if win32api.GetKeyState(win32con.VK_RBUTTON) < 0: # 右键
                ScanAndShot()
            else:
                time.sleep(0.08)
        except:
            pass
          
