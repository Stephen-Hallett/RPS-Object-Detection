import cv2 as cv
import numpy as np
import time

print(cv.__version__)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.35
CONFIDENCE_THRESHOLD = 0.5

def detect(image,net):
    blob = cv.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def load_capture():
    cap = cv.VideoCapture(0)
    return cap

class_list = ["Paper", "Rock", 'Scissors']

def wrap_detection(input_image,output_data):
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
    col,row, _ = frame.shape
    _max = max(col,row)
    result = np.zeros((_max,_max,3),np.uint8)
    result[0:col,0:row] = frame
    return result

def get_frame(cap):
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        cap.release()
        cv.destroyAllWindows()
        exit()
    return frame

def show_boxes(frame,zip):
    for (classid, confidence, box) in zip:
         color = colours[int(classid) % len(colours)]
         cv.rectangle(frame, box, color, 2)
         cv.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
         cv.putText(frame, "{} {:.2f}".format(class_list[classid], confidence), (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))




colours = [(255, 255, 0), (0, 255, 0), (0, 255, 255)]


net = cv.dnn.readNet("yolov5/runs/train/yolov5s_results_batch_sz64/weights/best.onnx")


cap = load_capture()
if not cap.isOpened():
    print("Cannot open camera")
    exit()

start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1



while True:
    frame = get_frame(cap)
    input_img = format_yolov5(frame)
    outs = detect(input_img,net)
    class_ids,confidences,boxes = wrap_detection(input_img, outs[0])
    show_boxes(frame, zip(class_ids, confidences, boxes))
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# print("Total frames: " + str(total_frames))