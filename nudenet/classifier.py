import tensorflow as tf

import os
import pydload
import numpy as np

from .image_utils import load_images

class Classifier:
    """
        Class for loading model and running predictions.
        For example on how to use take a look the if __name__ == '__main__' part.
    """

    nsfw_model = None

    def __init__(self):
        """
            model = Classifier()
        """
        url = "https://github.com/bedapudi6788/NudeNet/releases/download/v0/classifier_model"
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, ".NudeNet/")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        model_path = os.path.join(model_folder, "classifier")

        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        self.nsfw_model = tf.keras.models.load_model(model_path)

    def classify(
            self,
            image_paths=[],
            image_names=None,
            batch_size=4,
            image_size=(256, 256),
            categories=["unsafe", "safe"],
    ):
        """
            inputs:
                image_paths: list of image paths or can be a string too (for single image)
                batch_size: batch_size for running predictions
                image_size: size to which the image needs to be resized
                categories: since the model predicts numbers, categories is the list of actual names of categories
        """
        array_with_image_paths = []
        if isinstance(image_paths, str) or type(image_paths).__module__ == np.__name__:
            array_with_image_paths.append(image_paths)

        print('Get loaded image...')
        loaded_images, loaded_image_paths = load_images(
            array_with_image_paths, image_size, image_names
        )

        print("Start classifying...")
        model_preds = self.nsfw_model.predict(
            loaded_images, batch_size=batch_size
        )
        print("Classifying has finished...")
        preds = np.argsort(model_preds, axis=1).tolist()

        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(model_preds[i][pred])
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        images_preds = {}

        for i, loaded_image_path in enumerate(loaded_image_paths):
            if not isinstance(loaded_image_path, str):
                loaded_image_path = i

            images_preds[loaded_image_path] = {}
            for _ in range(len(preds[i])):
                images_preds[loaded_image_path][preds[i][_]] = probs[i][_]

        return images_preds


if __name__ == "__main__":
    print(
        '\n Enter path for the keras weights, leave empty to use "./nsfw.299x299.h5" \n'
    )
    weights_path = input().strip()
    if not weights_path:
        weights_path = "../nsfw.299x299.h5"

    m = Classifier()

    while 1:
        print(
            "\n Enter single image path or multiple images seperated by || (2 pipes) \n"
        )
        images = input().split("||")
        images = [image.strip() for image in images]
        print(m.predict(images), "\n")
