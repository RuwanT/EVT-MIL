import keras.backend as K
margin = 1.


# NOTE: loss has to return a tensor with the same number of elements as the batch size

def siamese_loss(y_true, y_pred):

    y_true_ = K.argmax(y_true, axis=-1)
    y_true_even = y_true_[::2]
    y_true_even = K.expand_dims(y_true_even, axis=-1)
    y_true_odd = y_true_[1::2]
    y_true_odd = K.expand_dims(y_true_odd, axis=-1)
    y_true_ = K.concatenate([y_true_even, y_true_odd], axis=-1)
    # label = 1 if dissimilar
    labels = K.sum(y_true_, axis=-1, keepdims=False) - 2 * K.prod(y_true_, axis=-1, keepdims=False)
    labels = K.cast(labels, dtype=K.floatx())

    y_pred_even = y_pred[::2, :]
    y_pred_odd = y_pred[1::2, :]

    l2error = K.sqrt(K.maximum(K.sum(K.square(y_pred_even - y_pred_odd), axis=-1, keepdims=False), K.epsilon()))

    contrastive_loss = 0.5*((1-labels) * K.square(l2error) + labels * K.square(K.maximum(margin - l2error, 0)))
    contrastive_loss = K.expand_dims(contrastive_loss, axis=-1)

    contrastive_loss_full = K.concatenate([contrastive_loss, contrastive_loss], axis=-1)
    contrastive_loss_full = K.reshape(contrastive_loss_full, shape=(-1, 1))
    contrastive_loss_full = K.squeeze(contrastive_loss_full, axis=-1)

    return contrastive_loss_full