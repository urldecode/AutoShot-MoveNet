
#from turtle import right, width
import cv2
import tensorflow as tf
import numpy as np
#import threading
import win32api
import win32gui
import win32ui
import win32con
import time
import numpy
import random
#bind MOUSE2 "+speed;r_cleardecals"
#慢走清除血迹

#避免枪口上抬，下压像素8值
#drop= 5
#两发子弹间隔时间-
    #awp                 0.13        1           
    #AK47       0.1      0.08        5-10        
    #m16A4      0.075    < 0.07      5           16
    #m16A4-2    0.075    0           5-15
    #沙鹰        0.222
#shottime = 0.05

# 检测值，置信度 
keypoint_threshold = 0.31
#bind MOUSE2 "+speed;r_cleardecals"

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

        #header_center = keypoints_all[0][3]
        # header_center = keypoints_all[0][1]

        left_should = keypoints_all[0][3]
        right_should = keypoints_all[0][4]
        # 0: 鼻子、1: 左眼、2: 右眼、3: 左耳、4: 右耳、5: 左肩、6: 右肩、7: 左肘、8: 右肘、9: 左腕、10: 右腕、11: 左胯、12: 右胯、13: 左膝、14: 右膝、15: 左踝、16: 右踝。
       #https://inews.gtimg.com/newsapp_bt/0/13809361254/1000

        header_center = (left_should + right_should) /2

        header_center = header_center.astype(np.int16).tolist()
        Shot_X = header_center[0] - ScreenShot_Center[0] 
        Shot_Y = header_center[1] - ScreenShot_Center[1] 
        

        #print("置信度：",'%.3f' % kpts_scores[3],"\t相对坐标：",header_center,"\t鼠标移动坐标：",Shot_X,Shot_Y)
        # ,end="" 
        #print(keypoints_all  )
 
interpreter = tf.lite.Interpreter(model_path="./lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
interpreter.allocate_tensors()


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.Session(config=config)

def movenet(input_image):
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def ScreenShot():
    #time1= time.time()
    #获取桌面
    hdesktop = win32gui.GetDesktopWindow()

    #返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
    desktop_dc = win32gui.GetWindowDC(hdesktop)

    #创建设备描述表
    img_dc = win32ui.CreateDCFromHandle(desktop_dc)
    #创建内存设备描述表
    mem_dc = img_dc.CreateCompatibleDC()
    #创建位图对象准备保存图片
    screenshot = win32ui.CreateBitmap()
    #为bitmap开辟存储空间
    screenshot.CreateCompatibleBitmap(img_dc, width, height)
    #将截图保存到screenshot中
    mem_dc.SelectObject(screenshot)

    #保存bitmap到内存设备描述表
    mem_dc.BitBlt((0, 0), (width, height), img_dc, (Zero_X, Zero_Y), win32con.SRCCOPY)

    ###保存bitmap到文件
    #screenshot.SaveBitmapFile(mem_dc, 'screenshot.bmp')

    ###获取位图信息
    # bmpinfo = screenshot.GetInfo()
    # print(bmpinfo)
    bmpstr = screenshot.GetBitmapBits(True)
    ###生成图像
    # im_PIL = Image.frombuffer('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr,'raw','BGRX',0,1)
    # print(im_PIL)
    # time2= time.time()
    # print ("截图时间:\t" , time2 - time1)

    im_opencv = numpy.frombuffer(bmpstr, dtype = 'uint8')
    im_opencv.shape = (height, width, 4)
    image = cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2RGB)

    
    # time3=time.time()
    # print ("图像转换时间:\t" ,time3 - time2)
    #显示图像
    # cv2.imshow("im_opencv",im_opencv) #显示
    # cv2.waitKey(0)

    # 释放内存DC
    img_dc.DeleteDC()
    mem_dc.DeleteDC()
    
    # 删除位图
    win32gui.DeleteObject(screenshot.GetHandle())

    #释放屏幕DC
    win32gui.ReleaseDC(hdesktop,desktop_dc)
    # time4= time.time()
    # print ("释放内存时间:\t" ,time4 - time3)

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
        #移动鼠标
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,Shot_X,Shot_Y+drop,0,0)
        #射击
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
        # 延迟
        shottime=round(random.uniform(0.04,0.07), 3)
        time.sleep(shottime)   
        time2 = time.time()
        print("检测到射击时间:\t", int((time2 - time1)*1000), "ms\t 下压:\t",drop , "\t 射击 间隔: \t", int(shottime*1000),"ms")
 

if __name__ =="__main__":
    while True:
        try:
            #if  win32api.GetKeyState(win32con.VK_MBUTTON) < 0:  # 中键
            if win32api.GetKeyState(win32con.VK_RBUTTON) < 0: # 右键
            #if win32api.GetKeyState(0x40)  < 0: #侧键2
                
                ScanAndShot()
                
            else:
                time.sleep(0.08)


        except:
            
            pass
          
