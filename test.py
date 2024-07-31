import numpy as np
import cv2
from glob import glob

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import *

def cal_iou(y_true, y_pred):
    x1 = max(y_true[0], y_pred[0])
    y1 = max(y_true[1], y_pred[1])
    x2 = min(y_true[2], y_pred[2])
    y2 = min(y_true[3], y_pred[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    true_area = (y_true[2] - y_true[0] + 1) * (y_true[3] - y_true[1] + 1)
    bbox_area = (y_pred[2] - y_pred[0] + 1) * (y_pred[3] - y_pred[1] + 1)

    iou = intersection_area / float(true_area + bbox_area - intersection_area)
    return iou

if __name__ == "__main__":
    
    model = tf.keras.models.load_model('/kaggle/working/best_model.keras', custom_objects={"ciou_loss":ciou_loss})

    print(f"Test : {len(test_images)} - {len(test_bboxes)} - {len(test_labels)}")

    mean_iou = []
    pred_labels = []
    images = []

    for image, true_bbox, true_labels in tqdm(zip(test_images, test_bboxes, test_labels), total=len(test_images)):
        name = image.split("/")[-1]

        image = cv2.imread(image, cv2.IMREAD_COLOR)
        x = cv2.resize(image, (width, height))
        x = (x - 127.5) / 127.5
        x = np.expand_dims(x, axis=0)
        
        true_x1, true_y1, true_x2, true_y2 = true_bbox

        pred_bbox, label = model.predict(x, verbose=0)
        pred_bbox = pred_bbox[0]
#         print(pred_bbox)
        label_index = np.argmax(label[0])
        pred_labels.append(label_index)
#         print(label_index)
#         print(true_labels)

        pred_x1 = int(pred_bbox[0] * image.shape[1])
        pred_y1 = int(pred_bbox[1] * image.shape[0])
        pred_x2 = int(pred_bbox[2] * image.shape[1])
        pred_y2 = int(pred_bbox[3] * image.shape[0])

        iou = cal_iou(true_bbox, [pred_x1, pred_y1, pred_x2, pred_y2])
        mean_iou.append(iou)

        image = cv2.rectangle(image, (true_x1, true_y1), (true_x2, true_y2), (255, 0, 0), 2) ## BLUE
        image = cv2.rectangle(image, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 2) ## RED
        
        font_size = 1.5
        pred_class_name = [i for i in class_dict if class_dict[i]== int(label_index)]
        cv2.putText(image, str(pred_class_name[0]) + ' '+ '(pred)', (80, 300), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)

        font_size = 1.5
        true_class_name = [i for i in class_dict if class_dict[i]== int(true_labels)]

        cv2.putText(image, str(true_class_name[0])+ ' '+ '(true)', (80, 350), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)

        font_size = 1.5
        cv2.putText(image, f"IoU: {iou:.4f}", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)
        images.append(image)

    score = np.mean(mean_iou, axis=0)
    mean_acc = accuracy_score(test_labels, pred_labels)
    print(f"Mean IoU: {score:.4f} - Acc: {mean_acc:.4f}")
