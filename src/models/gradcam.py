"""Grad-CAM (Block 10) — extracted from notebooks/03_transfer_learning.ipynb
into a reusable module so the Streamlit app can call the same logic that
was prototyped and validated there, rather than re-deriving it.

Two additions beyond the notebook's inline version, both because the
notebook always had `base_model` and `last_conv_layer_name` sitting in
its own local scope from training, whereas the app only has the loaded
`.keras` file:

- find_last_conv_layer(): re-derives the last 4D-output layer by name.
- get_base_model(): pulls the EfficientNetB0 submodel out of the loaded
  Sequential model. Assumes the saved architecture is exactly
  `Sequential([base_model, Dropout, Dense])` — i.e. base model at index
  0, matching how models/efficientnetb0_finetuned.keras was built. If
  that architecture changes, this assumption needs revisiting.

make_gradcam_heatmap() and overlay_gradcam() themselves are otherwise
unchanged from the notebook.
"""

import cv2
import numpy as np
import tensorflow as tf


def find_last_conv_layer(base_model):
    """Walk backward through base_model's layers and return the name of
    the last one with a 4D output (batch, h, w, channels) — the last
    convolutional-style layer, used as the Grad-CAM target layer.
    """
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except AttributeError:
            continue
    raise ValueError("No 4D-output (convolutional) layer found in base_model.")


def get_base_model(full_model):
    """Extract the EfficientNetB0 base submodel from the full model.

    Assumes `full_model` is `Sequential([base_model, Dropout, Dense])`,
    as saved by notebooks/03_transfer_learning.ipynb — the base model is
    layer 0.
    """
    return full_model.layers[0]


def make_gradcam_heatmap(img_array, model, base_model=None, last_conv_layer_name=None):
    """Compute a Grad-CAM heatmap for img_array's top predicted class.

    img_array: a single preprocessed image, batched (shape (1, H, W, 3)),
        using the SAME preprocessing the model was trained with
        (EfficientNet's preprocess_input — not /255.0).
    model: the full loaded model (base_model + Dropout + Dense).
    base_model, last_conv_layer_name: auto-derived via get_base_model()
        / find_last_conv_layer() if not supplied.

    Returns (heatmap, pred_index): heatmap is a 2D numpy array in [0, 1],
    pred_index is the model's predicted class for this image.
    """
    if base_model is None:
        base_model = get_base_model(model)
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(base_model)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_predictions = grad_model(img_array)
        # Pass the base model's pooled output through the rest of the
        # model (dropout + dense) to get final class predictions.
        predictions = model.layers[-1](model.layers[-2](base_predictions))
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy(), int(pred_index)


def overlay_gradcam(img_array, heatmap, img_size=224, alpha=0.4):
    """Overlay a Grad-CAM heatmap on top of a display image.

    img_array: a single image as a numpy array. Can be either the raw
        0-255 RGB image (recommended — pass the original upload resized
        to img_size, not the EfficientNet-preprocessed tensor, for a
        visually accurate overlay) or an already-preprocessed array; a
        min-max rescale to 0-255 is applied either way, so a raw 0-255
        input just gets rescaled onto itself.
    heatmap: 2D array in [0, 1], as returned by make_gradcam_heatmap().
    img_size: resize target for the heatmap to match img_array.
    alpha: heatmap opacity in the blend.

    Returns a uint8 RGB numpy array ready for display.
    """
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_display = img_array.numpy() if hasattr(img_array, "numpy") else np.asarray(img_array)
    img_display = img_display - img_display.min()
    max_val = img_display.max()
    img_display = (img_display / max_val * 255).astype(np.uint8) if max_val > 0 else img_display.astype(np.uint8)

    return cv2.addWeighted(img_display, 1 - alpha, heatmap_colored, alpha, 0)
