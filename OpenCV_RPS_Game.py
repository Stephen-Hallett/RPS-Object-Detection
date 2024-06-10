from RPS import RPS
import cv2 as cv
import time
import os
import numpy as np

"""
define input width and height of images for the model, also define the NMS threshold which determines the sensitivity for how many
boxes are drawn, since if the threshold is zero it will draw multiple boxes around the same object. Lastly also define the confidence
threshold which determines the minimum confidence that the model needs to have for an object class for it to show it on the canvas.
"""
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
NMS_THRESHOLD = 0.35
CONFIDENCE_THRESHOLD = 0.5

def detect(image,net):
    """
    Takes image from openCV, converts to RBG, normalises it (divides pixel numbers by 255) and passes it to the model
    in the form of a blob and returns predictions
    """
    blob = cv.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def load_capture(): #start OpenCV video stream
    cap = cv.VideoCapture(0)
    return cap

class_list = ["Paper", "Rock", 'Scissors']

def wrap_detection(input_image,output_data):
    """
    input_image is the current frame in the openCV canvas, output data for this function is the predictions gathered from the detect() function.
    This function takes the predictions of a bunch of boxes the model predicts there is an object in, remaps the boxes onto the shape of the opencv
    frame, and then returns only the boxes which meet the confidence and NMS thresholds.
    """
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width/INPUT_WIDTH
    y_factor = image_height/INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:
            classes_scores = row[5:]
            _, _, _, max_indx = cv.minMaxLoc(classes_scores)
            class_id = max_indx[1]

            if classes_scores[class_id] > 0.25:
                confidences.append(confidence)
                class_ids.append(class_id)
                x,y,w,h = row[0].item(), row[1].item(), row[2].item(), row[3].item()

                left = int((x-0.5*w)*x_factor)
                top = int((y-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                boxes.append(box)
    
    indexes = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids,result_confidences,result_boxes

def format_yolov5(frame):
    """
    yolo model was trained on square 640x640 data, so this formatting function makes it so if the opencv canvas is rectangular,
    black screen will be added above/next to the canvas before the model looks at the image to make predictions.
    """
    col,row, _ = frame.shape
    _max = max(col,row)
    result = np.zeros((_max,_max,3),np.uint8)
    result[0:col,0:row] = frame
    return result

def get_frame(cap):
    """
    does a small amount of preprocessing/checking before showing the frame
    """
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        cap.release()
        cv.destroyAllWindows()
        exit()
    return frame

def show_boxes(frame,zip):
    """
    Takes a zip() of boxes and overlays them on the openCV canvas.
    """
    for (classid, confidence, box) in zip:
         color = colours[int(classid) % len(colours)]
         cv.rectangle(frame, box, color, 2)
         cv.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
         cv.putText(frame, "{} {:.2f}".format(class_list[classid], confidence), (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))


symbol_images = {'r':(cv.imread(os.path.join('symbols','rock.png'))),
                'p':(cv.imread(os.path.join('symbols','paper.png'))),
                's':(cv.imread(os.path.join('symbols','scissors.png')))}

#STARTS THE VIDEO CAPTURE
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
start = (int(WIDTH/2-150),HEIGHT-240)
end = (int(WIDTH/2+150),HEIGHT)

"""
For each of RPS, there is a PNG corresponding to it which is shown on screen when the computer shows its prediction.
Below is code to take record of all the pixels in these pngs which are not black, so that when the png is shown there
isnt a huge black box around it
"""
img_pixels = {}
for symbol in ['r','p','s']:
    img_box = np.full((200,165,3),np.array([0,0,0]))
    img_box = cv.resize(img_box, (165,200))
    size = symbol_images[symbol].shape
    height,width = (200,int(200/(size[0]/size[1])))
    img = cv.resize(symbol_images[symbol],(width,height))
    img_box[0:200,int((165-width)/2):int((165-width)/2)+width] = img
    img_pixels[symbol] =[(i,j) for i in range(200) for j in range(165) if not np.array_equal(img_box[i][j],np.array([0,0,0]))]
    symbol_images[symbol] = img_box


def overlay(symbol,frame):
    """
    This overlays the RPS pngs given their recorded pixel values, allowing the rest of the pixel values
    around it to be equal to those of the frame, effectively just removing the background of the images.
    """
    if symbol in symbols:
        for i,j in img_pixels[symbol]:
            frame[i+876][j+880] = symbol_images[symbol][i][j]
        return frame

def show_score(frame, score):
    """
    Shows a score box at the top of the screen with user and computer scores
    """
    user_score_loc = (start[0]+(15 + (len(str(score[0])) % 2) * 45),80) #(start[0]+60,80)
    computer_score_loc = (start[0]+185,80)
    cv.rectangle(frame, (start[0],0),(end[0],125), (255,255,255), 2)
    cv.putText(frame,'-',(start[0]+115,80),cv.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),6,cv.LINE_AA)
    cv.putText(frame,str(score[0]),user_score_loc,cv.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),6,cv.LINE_AA)
    cv.putText(frame,str(score[1]),computer_score_loc,cv.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),6,cv.LINE_AA)

def show_prediction(frame,prediction):
    """
    Show computers move on the frame
    """
    cv.rectangle(frame, start,end, (255,255,255), 2)
    cv.rectangle(frame, start, (end[0],start[1]+30), (255,255,255), -1)
    cv.putText(frame,'I choose:',((int(WIDTH/2-140),HEIGHT-215)),font,fontScale,(0,0,0),2,cv.LINE_AA)
    frame = overlay(prediction,frame)


font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 4
colour = (0,255,0)
now = 0
centre = (int(WIDTH/2)-80,int(HEIGHT/2)+80)

score_width = start[0] + 15
score_height = 85
colours = [(255, 255, 0), (0, 255, 0), (0, 255, 255)]

symbols = ['r','p','s']

net = cv.dnn.readNet("objectdetection/yolov5s_results_batch_sz64_augmented4/weights/best.onnx")
rps = RPS(input("What is your name? "))

long = {'r': 'Rock',
        'p': 'Paper',
        's': 'Scissors'}

short = {'Rock': 'r',
        'Paper': 'p',
        'Scissors': 's'}

recorded = False

while True:
    #start by getting the frame from the camera and showing score
    frame = get_frame(cap)
    show_score(frame,rps.score)

    """
    When the user presses the space key to start the game, the "now" variable is recorded to the time at which they
    pressed the key. The code below allows for model to start making predictions on the frames for 7 seconds after
    the space key is pressed, and in that time having a three second countdown for the user to make their move,
    and for the computer to show their move for 3 seconds, and then record the result
    """
    if time.time() < (now+7):
        input_img = format_yolov5(frame)
        outs = detect(input_img,net)
        class_ids,confidences,boxes = wrap_detection(input_img, outs[0])
        show_boxes(frame, zip(class_ids, confidences, boxes))
        if time.time() < (now+1):
            cv.putText(frame,'3',centre,font,8,colour,thickness,cv.LINE_AA)
        elif time.time() < (now+2):
            cv.putText(frame,'2',centre,font,8,colour,thickness,cv.LINE_AA)
        elif time.time() < (now + 3):
            cv.putText(frame,'1',centre,font,8,colour,thickness,cv.LINE_AA)
        elif time.time() < (now + 6) and recorded == False:
            if len(class_ids) > 0:
                prediction = rps.predict()
                show_prediction(frame,prediction)
                user_played = short[class_list[class_ids[0]]]
                result = rps.played(user_played,prediction)
                recorded = True
                user_played = None
        elif time.time() < (now + 6) and recorded == True:
            show_prediction(frame,prediction)
            if result == 'l':
                cv.putText(frame,'You Lose!',(centre[0]-500,centre[1]),font,8,colour,thickness,cv.LINE_AA)
            elif result == 'w':
                cv.putText(frame,'You Win!',(centre[0]-450,centre[1]),font,8,colour,thickness,cv.LINE_AA)
            elif result == "d":
                cv.putText(frame,'We Draw!',(centre[0]-450,centre[1]),font,8,colour,thickness,cv.LINE_AA)
            else:
                cv.putText(frame,'Unrecognised',(centre[0]-550,centre[1]),font,6,colour,thickness,cv.LINE_AA)
        elif time.time() < (now + 6) and recorded == False:
            cv.putText(frame,'Unrecognised',(centre[0]-550,centre[1]),font,6,colour,thickness,cv.LINE_AA)
        else:
            recorded = False

    
    cv.imshow('RPS',frame)

    #Below will start the game process if user presses space, or quit the program if they press "Q"
    if cv.waitKey(1) == ord(" "):
        if time.time() > (now+7):
            now = time.time()
    elif cv.waitKey(1) == ord('q'):
        break

rps.quit()
cap.release()
cv.destroyAllWindows()