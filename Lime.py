import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lime import lime_image

from tensorflow.keras.applications.resnet50 import (ResNet50, preprocess_input, decode_predictions)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from scipy.ndimage import label



model = tf.keras.models.load_model("Models/FPSwinPAM_Extractor.h5")


def predict_fn(batch_images):

    batch_images = np.array(batch_images)

    if batch_images.shape[-1] == 3:
        batch_images = batch_images.mean(axis=-1)

    batch_images = batch_images.astype(np.float32)
    batch_images = np.expand_dims(batch_images, axis=-1)
    
    pred = model.predict(batch_images, verbose=0)
    return pred



def lime_explainer(image):
    
    explainer = lime_image.LimeImageExplainer()
    image_rgb = np.stack([image, image, image], axis=-1)
    
    explanation = explainer.explain_instance(
        image_rgb,
        classifier_fn=predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=1000 
    )
    
    total_superpixels = len(np.unique(explanation.segments))
    print("Total superpixels generated:", total_superpixels)
    
    top_label = explanation.top_labels[0]
    
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    
    feature_weights = explanation.local_exp[top_label]
    feature_weights = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)
    
    labels = [f"SP {i}" for i, _ in feature_weights]
    weights = [w for _, w in feature_weights]
    
    plt.figure(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
    
    plt.barh(labels, weights, color=colors)
    plt.gca().invert_yaxis()
    plt.title("LIME Superpixel Feature Importance", fontname="Times New Roman",fontweight='bold',fontsize=20)
    plt.xticks(fontname="Times New Roman",fontweight='bold',fontsize=18)
    plt.yticks(fontname="Times New Roman",fontweight='bold',fontsize=18)
    plt.xlabel("Contribution", fontname="Times New Roman",fontweight='bold',fontsize=20)
    plt.show()






def img_lime(img):
    
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    
    elif len(img.shape) == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    
    img = resize(
        img,
        (224, 224),
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)
    
    disp_img = img.copy()
    disp_img -= disp_img.min()
    
    if disp_img.max() > 0:
        disp_img /= disp_img.max()
    
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )
    
    x = GlobalAveragePooling2D()(base_model.output)
    
    feature_layer = Dense(512, activation='relu', name='Feature_512')(x)
    output_layer = Dense(1000, activation='softmax')(feature_layer)
    model = Model(inputs=base_model.input, outputs=output_layer)



    def predict_fn(images):
        images = np.array(images)
        images = preprocess_input(images.copy())
        return model.predict(images, verbose=0)
    
    
    input_img = np.expand_dims(img, axis=0)
    predictions = predict_fn(input_img)
    
    top_class = np.argmax(predictions)
    
    explainer = lime_image.LimeImageExplainer()
    
    explanation = explainer.explain_instance(
        image=img.astype(np.double),
        classifier_fn=predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        label=top_class,
        positive_only=True,
        num_features=15,
        hide_rest=False
    )
    
    gray = np.mean(disp_img, axis=-1)
    roi_mask = gray > 0.15
    
    labeled, num = label(roi_mask)
    
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
    
        largest_component = np.argmax(sizes)
        roi_mask = (labeled == largest_component)
    
    
    mask = mask * roi_mask.astype(np.uint8)
    
    labeled_mask, num_mask = label(mask)
    clean_mask = np.zeros_like(mask)
    
    for i in range(1, num_mask + 1):
    
        region = (labeled_mask == i)
    
        if np.sum(region) > 100:
            clean_mask[region] = 1
    
    mask = clean_mask
    
    temp, mask = explanation.get_image_and_mask(
        label=top_class,
        positive_only=False,
        hide_rest=False,
        num_features=20,
        min_weight=0.0
    )
    
    overlay = disp_img.copy()
    # positive -> green
    overlay[mask == 1] = [0, 1, 0]
    # negative -> red
    overlay[mask == -1] = [1, 0, 0]
    
    lime_img = mark_boundaries(
        overlay,
        mask != 0,
        color=(1,1,0),
        mode='thick'
    )
    
    plt.figure(figsize=(6,6))
    plt.imshow(lime_img)
    plt.title("LIME Explanation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()











