import os
import json
import random


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} not exist.".format(root)

    # Traversal
    ear_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sort
    ear_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(ear_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    train_image_path = []
    train_image_label = []
    val_image_path = []
    val_image_label = []
    every_class_num = []
    supported = ['.png', '.PNG']
    for cla in ear_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # print(images)
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        images_without_r = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if (os.path.splitext(i)[-1] in supported) and ('R' not in i) and ('r' not in i)]
        # print("images_without_r:")
        # print(images_without_r)
        val_path = random.sample(images_without_r, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_image_path.append(img_path)
                val_image_label.append(image_class)
            else:
                train_image_path.append(img_path)
                train_image_label.append(image_class)
    print("{} images were found in the dataset".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_image_path)))
    print("{} images for validation".format(len(val_image_path)))
    return train_image_path, train_image_label, val_image_path, val_image_label
