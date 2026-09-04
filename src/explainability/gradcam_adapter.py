"""Grad-CAM adapter (SIH26038 Phase 6).

Two paths, one contract — returns (heatmap_0..1_2D, info_dict):
1. Predicted-class on the EfficientNetB0 baseline: delegates to the
   ORIGINAL Phase 1 make_gradcam_heatmap() UNCHANGED (semantics preserved).
2. Explicit target class (any model): generic gradient path reusing the
   Phase 1 layer helpers (get_base_model/find_last_conv_layer) where they
   apply, with a model-wide fallback scan otherwise.

Records explained_class, its probability, target layer, heatmap shape and
normalization. Output is an ACTIVATION heatmap, never a probability map.
"""

import numpy as np


def _normalise(h, method="minmax"):
    h = np.asarray(h, dtype=float)
    if method == "minmax":
        lo, hi = h.min(), h.max()
        return np.zeros_like(h) if hi <= lo else (h - lo) / (hi - lo)
    raise ValueError(f"Unknown heatmap normalization: {method}.")


def _is_efficientnet_like(model):
    """Original Phase 1 path requires Sequential([submodel, ...]) where
    layers[0] is itself a Model (the EfficientNetB0 base). Anything else
    uses the generic targeted path — even for the predicted class."""
    try:
        from tensorflow.keras.models import Model

        return isinstance(model.layers[0], Model)
    except (AttributeError, IndexError):
        return False


def _ensure_built(model, sample):
    """Loaded Sequential models may lack .input until functionally invoked.
    Re-wrap (shared weights, identical outputs) when needed."""
    import tensorflow as tf

    try:
        _ = model.input
        return model
    except AttributeError:
        inp = tf.keras.Input(shape=tuple(sample.shape[1:]), name="explain_input")
        return tf.keras.Model(inp, model(inp))


def _find_last_conv(model):
    """Deepest 4D-output layer object, descending into nested submodels
    (e.g. EfficientNetB0 base inside the Sequential grader)."""
    from tensorflow.keras.models import Model

    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            try:
                return _find_last_conv(layer)
            except ValueError:
                continue
        try:
            if len(layer.output.shape) == 4:
                return layer
        except (AttributeError, ValueError):
            continue
    raise ValueError("No 4D-output (convolutional) layer found.")


def _last_conv_name(model):
    return _find_last_conv(model).name


def _efficientnet_targeted(img_batch, model, target_class):
    """Targeted Grad-CAM through the Phase 1 graph construction (base model
    in/out + final Dropout/Dense push-through). Same graph the validated
    predicted-class path uses; only the class channel differs."""
    import tensorflow as tf

    from src.models.gradcam import find_last_conv_layer, get_base_model

    base = get_base_model(model)
    lname = find_last_conv_layer(base)
    grad_model = tf.keras.models.Model(
        inputs=base.input,
        outputs=[base.get_layer(lname).output, base.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, base_predictions = grad_model(img_batch)
        predictions = model.layers[-1](model.layers[-2](base_predictions))
        class_channel = predictions[:, int(target_class)]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_outputs[0] @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0)
    peak = tf.math.reduce_max(heat)
    heat = heat / peak if float(peak) > 0 else heat
    return np.asarray(heat, dtype=float), lname


def targeted_gradcam(img_batch, model, target_class, layer=None):
    """Generic Grad-CAM for an explicit target class. img_batch: (1,H,W,3)
    model-ready tensor. layer: layer object or name (auto: deepest conv).
    Returns (heatmap, target_class)."""
    import tensorflow as tf

    conv = layer if not isinstance(layer, (str, type(None))) else None
    if conv is None:
        name = layer or _find_last_conv(model).name
        try:
            conv = model.get_layer(name)
        except ValueError:
            conv = _find_last_conv(model)
    graph = _ensure_built(model, img_batch)
    grad_model = tf.keras.models.Model(
        inputs=graph.input,
        outputs=[conv.output, graph.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        channel = preds[:, int(target_class)]
    grads = tape.gradient(channel, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
    heat = tf.maximum(heat, 0)
    peak = tf.math.reduce_max(heat)
    heat = heat / peak if float(peak) > 0 else heat
    return np.asarray(heat, dtype=float), int(target_class)


def explain_class(img_batch, model, target_class=None, cfg=None):
    """Explain target_class (default: predicted argmax). Returns dict with
    heatmap (low-res 2D), explained_class, probability, layer, shape."""
    import tensorflow as tf  # noqa: F401 (backend presence check)

    cfg = cfg or {}
    probs = np.asarray(model.predict(img_batch, verbose=0)[0], dtype=float)
    target = int(np.argmax(probs)) if target_class is None else int(target_class)
    layer = (cfg.get("gradcam", {}).get("target_layer")
             if isinstance(cfg.get("gradcam"), dict) else None)
    used_original = target_class is None and layer is None and _is_efficientnet_like(model)
    if used_original:
        from src.models.gradcam import (find_last_conv_layer, get_base_model,
                                        make_gradcam_heatmap)

        base = get_base_model(model)
        lname = find_last_conv_layer(base)
        heat, _ = make_gradcam_heatmap(img_batch, model, base, lname)
    elif _is_efficientnet_like(model):
        heat, lname = _efficientnet_targeted(img_batch, model, target)
    else:
        heat, _ = targeted_gradcam(img_batch, model, target, layer=layer)
        lname = layer if isinstance(layer, str) else _find_last_conv(model).name
    heat = _normalise(heat, (cfg.get("gradcam", {}) or {}).get("normalization", "minmax"))
    if not np.isfinite(heat).all():
        raise ValueError("Non-finite Grad-CAM heatmap.")
    return {
        "heatmap": heat,
        "explained_class": target,
        "explained_class_probability": round(float(probs[target]), 4),
        "target_layer": lname,
        "heatmap_shape": list(heat.shape),
        "normalization": "minmax",
        "via_original_path": used_original,
    }
