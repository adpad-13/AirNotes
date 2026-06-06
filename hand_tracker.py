import cv2
import time
import math
# Load the HandLandmarker Model
import mediapipe as mp
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque

smooth_x = deque(maxlen=3)
smooth_y = deque(maxlen=3)

latest_result = None
prev_x = None
prev_y = None
def process_result(result,output_image,timestamp_ms):
    global latest_result
    latest_result = result
    
def crop_and_resize(canvas):
    gray_canvas = cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY)    
    rows , cols = np.nonzero(gray_canvas)
    if rows.size > 0 and cols.size > 0:  
        y_min = rows.min()
        y_max=rows.max()
        x_min = cols.min()
        x_max = cols.max()
        cropped_canvas = gray_canvas[y_min:y_max,x_min:x_max]
        resized_canvas = cv2.resize(cropped_canvas,(28,28)) 
        return resized_canvas


options  = vision.HandLandmarkerOptions(
    base_options = mp_base_options.BaseOptions(model_asset_path = 'hand_landmarker.task'),
    running_mode = vision.RunningMode.LIVE_STREAM,
    num_hands =1,
    min_hand_detection_confidence = 0.4,
    min_hand_presence_confidence  = 0.5,
    min_tracking_confidence = 0.3,
    result_callback = process_result
)
def main():
    
    global prev_x , prev_y , smooth_x , smooth_y
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        canvas = np.zeros((height,width,3),dtype=np.uint8)
        is_drawing = False
        last_draw_time = 0
        while True:
            global prev_x
            
            ret, frame = cap.read()
            frame=cv2.flip(frame,1)
            if not ret:  break
            rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            timestamp  = time.time_ns() // 1_000_000
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
            landmarker.detect_async(mp_image,timestamp)
            if latest_result is not None and latest_result.hand_landmarks:
                thumb = latest_result.hand_landmarks[0][4]
                index = latest_result.hand_landmarks[0][8]
                dist = math.sqrt((thumb.x-index.x)**2 + (thumb.y-index.y)**2) 
                
                if dist<0.025:
                    is_drawing = True
                elif dist> 0.045:
                    is_drawing = False
                    smooth_x.clear()
                    smooth_y.clear()
                    prev_x = None
                    prev_y = None
                if is_drawing:
                    last_draw_time = time.time()
                    curr_x = int(index.x*width)
                    curr_y = int(index.y*height)
                    smooth_x.append(curr_x)
                    smooth_y.append(curr_y)
                    avg_x = int(sum(smooth_x)/len(smooth_x))
                    avg_y = int(sum(smooth_y)/len(smooth_y))
                    if prev_x is not None:
                        cv2.line(canvas, (prev_x,prev_y),(avg_x,avg_y),(0,0,255),2)
                    prev_x = avg_x
                    prev_y= avg_y
            
            if not is_drawing and last_draw_time > 0:
                if(time.time() - last_draw_time)> 1.5:
                    char_image = crop_and_resize(canvas)
                    if char_image is not None:
                        print("character shape",char_image.shape)

                    canvas = np.zeros((height,width,3),dtype=np.uint8)

                    last_draw_time = 0


            new_frame = cv2.addWeighted(frame,1,canvas,1,0)
        
            cv2.imshow('camera feed',new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()