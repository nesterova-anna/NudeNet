import os
import cv2
import pydload
import tarfile
import logging
import numpy as np
import tensorflow as tf
from .detector_utils import preprocess_image


def dummy(x):
    return x


FILE_URLS = {
    "default": {
        "checkpoint": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_checkpoint_tf.tar",
        "classes": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_classes",
    },
    "base": {
        "checkpoint": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_base_checkpoint_tf.tar",
        "classes": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_base_classes",
    },
}


class Detector:
    detection_model = None
    classes = None

    def __init__(self, model_name="default"):
        """
            model = Detector()
        """
        checkpoint_url = FILE_URLS[model_name]["checkpoint"]
        classes_url = FILE_URLS[model_name]["classes"]

        home = os.path.expanduser("~")
        model_folder = os.path.join(home, f".NudeNet/{model_name}")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        checkpoint_tar_file_name = os.path.basename(checkpoint_url)
        checkpoint_name = checkpoint_tar_file_name.replace(".tar", "")

        self.checkpoint_path = os.path.join(model_folder, checkpoint_name)
        checkpoint_tar_file_path = os.path.join(model_folder, checkpoint_tar_file_name)
        classes_path = os.path.join(model_folder, "classes")

        if not os.path.exists(self.checkpoint_path):
            print("Downloading the checkpoint to", self.checkpoint_path)
            pydload.dload(
                checkpoint_url, save_to_path=checkpoint_tar_file_path, max_time=None
            )
            with tarfile.open(checkpoint_tar_file_path) as f:
                f.extractall(path=os.path.dirname(checkpoint_tar_file_path))
            os.remove(checkpoint_tar_file_path)

        if not os.path.exists(classes_path):
            print("Downloading the classes list to", classes_path)
            pydload.dload(classes_url, save_to_path=classes_path, max_time=None)

        self.classes = [c.strip() for c in open(classes_path).readlines() if c.strip()]

    def detect(self, img_path, mode="default", min_prob=None):
        print("Start detecting...")
        if mode == "fast":
            image, scale = preprocess_image(img_path, min_side=480, max_side=800)
            if not min_prob:
                min_prob = 0.5
        else:
            image, scale = preprocess_image(img_path)
            if not min_prob:
                min_prob = 0.6

        print("Detecting has finished...")
        sample = np.expand_dims(image, axis=0)

        with tf.compat.v1.Session() as sess:
            model = tf.compat.v1.saved_model.load(sess, ['serve'], self.checkpoint_path)
            x_name = model.signature_def['predict'].inputs['images'].name
            x = sess.graph.get_tensor_by_name(x_name)
            y1_name = model.signature_def['predict'].outputs['output1'].name
            y1 = sess.graph.get_tensor_by_name(y1_name)
            y2_name = model.signature_def['predict'].outputs['output2'].name
            y2 = sess.graph.get_tensor_by_name(y2_name)
            y3_name = model.signature_def['predict'].outputs['output3'].name
            y3 = sess.graph.get_tensor_by_name(y3_name)
            pred = sess.run([y1, y2, y3], feed_dict={x: sample})

        boxes, scores, labels = pred[0], pred[1], pred[2]
        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = self.classes[label]
            processed_boxes.append({"box": box, "score": score, "label": label})

        return processed_boxes

    def censor(self, img_path, out_path=None, visualize=False, parts_to_blur=[]):
        if not out_path and not visualize:
            print(
                "No out_path passed and visualize is set to false. There is no point in running this function then."
            )
            return

        image = cv2.imread(img_path)
        boxes = self.detect(img_path)

        if parts_to_blur:
            boxes = [i["box"] for i in boxes if i["label"] in parts_to_blur]
        else:
            boxes = [i["box"] for i in boxes]

        for box in boxes:
            part = image[box[1] : box[3], box[0] : box[2]]
            image = cv2.rectangle(
                image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), cv2.FILLED
            )

        if visualize:
            cv2.imshow("Blurred image", image)
            cv2.waitKey(0)

        if out_path:
            cv2.imwrite(out_path, image)


if __name__ == "__main__":
    m = Detector()
    print(m.detect("/Users/bedapudi/Desktop/n2.jpg"))
