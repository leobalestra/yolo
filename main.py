#darknet
import cv2
import time
import numpy as np

#Carregar cores aleatorias
COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255)]

#Carrega as classee pré-treinadas
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#captura de video 0 - Para webcam
cap = cv2.VideoCapture(0)

#Carrega os pesos da rede neural
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

#parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)


while(1):
    
    #capture o frame
    ret, frame = cap.read()

    #detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    for (classid, score, box) in zip(classes, scores, boxes):

        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid[0]]} : {score}"

        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Video", frame)
   
    # espera da resposta
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()