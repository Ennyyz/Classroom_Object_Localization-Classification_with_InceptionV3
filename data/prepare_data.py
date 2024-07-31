import tensorflow as tf
import numpy as np
import cv2
from glob import glob

global height
global width
global num_classes
global AUTOTUNE

def load_labels(data):
    names = data['class'].unique().tolist()
    return names

def load_data(path, data, classes, p_data=None):
    images = []
    bboxes = []
    labels = []
    
    
    for index, row in data.iterrows():
        name = row['filename']
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        label = int(row['class_encoded'])
        
        if p_data == 'train':
            image = os.path.join(path, "train", name)
        elif p_data == 'valid':
            image = os.path.join(path, "valid" , name)
        else:
            image = os.path.join(path, "test" , name)

        bbox = [x1, y1, x2, y2]

        images.append(image)
        bboxes.append(bbox)
        labels.append(label)

    return images, bboxes, labels

def read_image_bbox(path, bbox, label_index):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    image = cv2.resize(image, (width, height))
    image = (image - 127.5) / 127.5 ## [-1, +1]
    image = image.astype(np.float32)

    x1, y1, x2, y2 = bbox

    norm_x1 = float(x1/w)
    norm_y1 = float(y1/h)
    norm_x2 = float(x2/w)
    norm_y2 = float(y2/h)
    norm_bbox = np.array([norm_x1, norm_y1, norm_x2, norm_y2], dtype=np.float32)

    label = [0] * num_classes
    label[label_index] = 1 #Fix here
    class_label = np.array(label, dtype=np.float32)

    return image, norm_bbox, class_label

def parse(image, bbox, label):
    image, bbox, label = tf.numpy_function(read_image_bbox, [image, bbox, label], [tf.float32, tf.float32, tf.float32])
    image.set_shape((height, width, 3))
    bbox.set_shape((4))
    label.set_shape((num_classes))
    return (image), (bbox, label)

def tf_dataset(images, bboxes, labels, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((images, bboxes, labels))
    ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.cache().shuffle(1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    else:
        ds = ds.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)   
    return ds

def load_dataset(path, data_type, classes):
    
    train, valid, test = data_type[0], data_type[1], data_type[2]
    
    train_images, train_bboxes, train_labels = load_data(path, train, classes, 'train')
    
    valid_images, valid_bboxes, valid_labels = load_data(path, valid, classes, 'valid')
    
    test_images, test_bboxes, test_labels = load_data(path, test, classes, 'test')

    return (train_images, train_bboxes, train_labels), (valid_images, valid_bboxes, valid_labels), (test_images, test_bboxes, test_labels)

if __name__ == "__main__":

    height = 320
    width = 320
    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE
    
    data_type = [train, valid, test]
    path = '/kaggle/input/custom-dataset-new'
    
    classes = load_labels(train)
    num_classes = len(classes)
    
    (train_images, train_bboxes, train_labels), (valid_images, valid_bboxes, valid_labels), (test_images, test_bboxes, test_labels) = load_dataset(path, data_type, classes)

    train_ds = tf_dataset(train_images, train_bboxes, train_labels, batch_size, shuffle=True)
    valid_ds = tf_dataset(valid_images, valid_bboxes, valid_labels, batch_size )
        
    print(f"Classes: {classes}")
    print(f"Total class : {num_classes}")
    print(f"Train: {len(train_images)} - {len(train_bboxes)} - {len(train_labels)}")
    print(f"Valid: {len(valid_images)} - {len(valid_bboxes)} - {len(valid_labels)}")
    print(f"Test : {len(test_images)} - {len(test_bboxes)} - {len(test_labels)}")
