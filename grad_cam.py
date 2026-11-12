import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input



def gradcam(img, grad_model):

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.zeros(conv_outputs.shape[:2])

    for i,w in enumerate(weights):
        heatmap += w * conv_outputs[:,:,i]

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    
    return heatmap.numpy()




def gradcam_plot(image):
    
    grad_model = load_model("Models/FPSwinPAM_grad.h5")
    image_rgb = np.stack([image,image,image], axis=-1)
    
    img = tf.image.resize(image_rgb, (224,224)).numpy()
    input_img = preprocess_input(np.expand_dims(img, axis=0))
    
    heatmap = gradcam(input_img, grad_model)
    
    heatmap = cv2.resize(heatmap,
        (image_rgb.shape[1], image_rgb.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    original = image_rgb.astype(np.uint8)
    
    overlay = cv2.addWeighted(
        original, 0.55, heatmap_color, 0.45, 0)
    
    
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1,3,2)
    plt.imshow(
        heatmap,
        cmap="jet"
    )
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    
    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()



# images = np.load("Features/decrypted_images.npy",mmap_mode="r")
# gradcam_plot(images)






# base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
# base_model.trainable = False

# last_conv_layer = "conv5_block3_out"

# grad_model = tf.keras.models.Model(
#     inputs=base_model.input,
#     outputs=[
#         base_model.get_layer(last_conv_layer).output,
#         base_model.output
#     ]
# )

# grad_model.save("Models/FPSwinPAM_grad.h5")


















