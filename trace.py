import torch
import cv2
import torch.jit
model = torch.hub.load("ultralytics/yolov5","yolov5s")
img = 'data/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# Inference
results = model(img)
# Results
results.print()
print(results)
img =cv2.imread(img)
classNames = results.names
for pred in results.pred[0].tolist():
    x1,y1,x2,y2,score,classIndex = pred
    label = classNames[int(classIndex)]
    cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,255),2)
    cv2.putText(img,label,org=(int(x1),int(y1)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale= 1.2,color=(255,0,255),thickness=2)
cv2.imshow("zidane",img)
cv2.waitKey(0)
