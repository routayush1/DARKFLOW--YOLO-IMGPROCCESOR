import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format ="svg"
options={
    'model':'cfg/yolo.cfg',
    'load':'bin/yolo.weights',
    'threshold':0.1,
    'gpu':1.0
}

tfnet=TFNet(options)
img=cv2.imread("DOGE.JPG",cv2.IMREAD_COLOR)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

results=tfnet.return_predict(img)
print (results)
img.shape

for result in results:
    top_x = result['topleft']['x']
    top_y = result['topleft']['y']

    btm_x = result['bottomright']['x']
    btm_y = result['bottomright']['y']

    confidence = result['confidence']
    label = result['label'] + " " + str(round(confidence, 3))

    if confidence > 0.1:
        newImage = cv2.rectangle(img, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
        newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                               (0, 230, 0), 1, cv2.LINE_AA)




plt.imsave("bin/",img)
plt.show()




