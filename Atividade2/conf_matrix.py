import numpy as np
import matplotlib.pyplot as plt
import cv2
def plot_confusion_matrix(classifier, X, y):
    class_values = np.unique(y)
    n = len(class_values)
    conf_matrix = np.zeros((n,n), dtype=np.uint8)
    y_pred = classifier.predict(X)
    i=0
    for i in range(n):
        for j in range(n):
            conf_matrix[i,j] = np.sum((y == class_values[i]) & (y_pred == class_values[j]))
    
    rgb = cv2.cvtColor(conf_matrix, cv2.COLOR_BGR2RGB) 
    hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv_img[:,:,0] = 100
    hsv_img[:,:,1] = 255
    hsv_img[:,:,2] = conf_matrix.copy()
    matrix = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    norm_image = cv2.normalize(matrix, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.imshow(norm_image), plt.show()
    
    return conf_matrix