from keras import backend as K
import tensorflow as tf


def weighted_MASE(y_true, y_pred, horizon, decay_lambda):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    n = tf.shape(y_true)[0]
    e = tf.abs(tf.subtract(y_true, y_pred))
    max_e = tf.reduce_max(e)
    s_diff = tf.reduce_sum(tf.abs(tf.subtract(y_true[1:], y_true[:-1])), 0)
    d = 1 / tf.multiply(tf.cast(n - 1, tf.float32), s_diff)

    q = e / tf.expand_dims(d, 0)

    w = tf.tile(tf.range(horizon), [tf.cast(n / horizon, tf.int32)])
    w_q = q * tf.exp(-decay_lambda * tf.cast(w, tf.float32))

    wMASE = tf.reduce_mean(w_q, 0)

    return wMASE


def MASE_loss(y_true, y_pred):
    horizon = tf.shape(y_true)[1]
    y_t = tf.reshape(y_true, [-1])
    y_f = tf.reshape(y_pred, [-1])

    n = tf.shape(y_t)[0]
    e = tf.abs(tf.subtract(y_t, y_f))
    max_e = tf.reduce_max(e)
    s_diff = tf.reduce_sum(tf.abs(tf.subtract(y_t[1:], y_t[:-1])), 0)
    d = 1 / tf.multiply(tf.cast(n - 1, tf.float32), s_diff)

    q = e / tf.expand_dims(d, 0)

    return tf.reduce_mean(q, 0)


def PICP_loss(y_true, y_pred):
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:, ::3], [-1])
    y_l = tf.reshape(y_pred[:, 2::3], [-1])

    K_HU = tf.maximum(0., tf.sign(y_u - y_t))
    K_HL = tf.maximum(0., tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    return tf.reduce_mean(K_H)


def MPIW_c_loss(y_true, y_pred):
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:, ::3], [-1])
    y_l = tf.reshape(y_pred[:, 2::3], [-1])

    K_HU = tf.maximum(0., tf.sign(y_u - y_t))
    K_HL = tf.maximum(0., tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    return tf.reduce_sum(tf.multiply((y_u - y_l), K_H)) / (tf.reduce_sum(K_H) + 1)


def MPIW_b_loss(y_true, y_pred):
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:, ::3], [-1])
    y_l = tf.reshape(y_pred[:, 2::3], [-1])

    K_HU = tf.maximum(0., tf.sign(y_u - y_t))
    K_HL = tf.maximum(0., tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    C = tf.reduce_sum(K_H)

    return tf.reduce_mean(tf.multiply((y_u - y_l), K_H)) * C / tf.cast(tf.shape(K_H), tf.float32) + tf.reduce_mean(
        tf.abs(K_H - 1)) ** (C)


def qd_objective_lstm_b(y_true, y_pred, decay_lambda, soften_, lambda_, alpha_, n_):
    '''Loss_QD-soft, from algorithm 1'''
    h = tf.shape(y_true)[1]
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:, ::3], [-1])
    y_f = tf.reshape(y_pred[:, 1::3], [-1])
    y_l = tf.reshape(y_pred[:, 2::3], [-1])

    wMASE = weighted_MASE(y_t, y_f, h, decay_lambda)

    K_HU = tf.maximum(0., tf.sign(y_u - y_t))
    K_HL = tf.maximum(0., tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    K_SU = tf.sigmoid(soften_ * (y_u - y_t))
    K_SL = tf.sigmoid(soften_ * (y_t - y_l))
    K_S = tf.multiply(K_SU, K_SL)

    C = tf.reduce_sum(K_H)
    MPIW_b = tf.reduce_mean(tf.multiply((y_u - y_l), K_H)) * C / tf.cast(tf.shape(K_H), tf.float32) + tf.reduce_mean(
        tf.abs(K_H - 1)) ** (C)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)

    pinball = pinball_loss(y_t, y_f)

    Loss_S = MPIW_b + lambda_ * (
            n_ / (alpha_ * (1 - alpha_)) * tf.maximum(0., (1 - alpha_) - PICP_S) * 1 + 1 * wMASE)
    return Loss_S


def lube_berken_loss_b(y_true, y_pred, soften_, lambda_, alpha_, n_):
    '''Loss_QD-soft, from algorithm 1'''
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:, ::2], [-1])
    y_l = tf.reshape(y_pred[:, 1::2], [-1])

    K_HU = tf.maximum(0., tf.sign(y_u - y_t))
    K_HL = tf.maximum(0., tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    K_SU = tf.sigmoid(soften_ * (y_u - y_t))
    K_SL = tf.sigmoid(soften_ * (y_t - y_l))
    K_S = tf.multiply(K_SU, K_SL)

    C = tf.reduce_sum(K_H)
    MPIW_b = tf.reduce_mean(tf.multiply((y_u - y_l), K_H)) * C / tf.cast(tf.shape(K_H), tf.float32) + tf.reduce_mean(
        tf.abs(K_H - 1)) ** (C)
    PICP_S = tf.reduce_mean(K_S)

    Loss_S = MPIW_b + lambda_ * (
            n_ / (alpha_ * (1 - alpha_)) * tf.maximum(0., (1 - alpha_) - PICP_S) ** 2)
    return Loss_S


def lube_berken_loss_c(y_true, y_pred, soften_, lambda_, alpha_, n_):
    '''Loss_QD-soft, from algorithm 1'''
    y_t = tf.reshape(y_true, [-1])
    y_u = tf.reshape(y_pred[:, ::2], [-1])
    y_l = tf.reshape(y_pred[:, 1::2], [-1])

    K_HU = tf.maximum(0., tf.sign(y_u - y_t))
    K_HL = tf.maximum(0., tf.sign(y_t - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    K_SU = tf.sigmoid(soften_ * (y_u - y_t))
    K_SL = tf.sigmoid(soften_ * (y_t - y_l))
    K_S = tf.multiply(K_SU, K_SL)

    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l), K_H)) / (tf.reduce_sum(K_H) + 1)
    PICP_S = tf.reduce_mean(K_S)

    Loss_S = MPIW_c + lambda_ * (
            n_ / (alpha_ * (1 - alpha_)) * tf.maximum(0., (1 - alpha_) - PICP_S) ** 2)
    return Loss_S


def pinball_loss(y_true, y_pred, tau):
    pin = K.mean(K.maximum(y_true - y_pred, 0) * tau +
                 K.maximum(y_pred - y_true, 0) * (1 - tau))
    return pin


def msis_no_normalization(y_true, y_pred):
    y_t = tf.reshape(y_true, [-1, 48])
    y_u = tf.reshape(y_pred[:, ::2], [-1, 48])
    y_l = tf.reshape(y_pred[:, 1::2], [-1, 48])

    print(tf.shape(y_l))

    upper_wrong = tf.maximum(0., tf.sign(y_t - y_u))
    lower_wrong = tf.maximum(0., tf.sign(y_l - y_t))

    lower_error = tf.multiply(lower_wrong, (y_l - y_t))
    upper_error = tf.multiply(upper_wrong, (y_t - y_u))
    return  tf.reduce_mean(tf.reduce_sum(y_u - y_l + 40.0 * lower_error + 40.0 * upper_error, axis=1)/48)
