import tensorflow as tf

def loss_TPD(T, beta, l):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]
        d = y_true[:, 2]
        
        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))

        #BCE instance by instance
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_inst = bce(y_pred, y_pr )
        bce_r = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)
        return tf.reduce_mean((1-l)*(bce_r) + l*(tf.math.multiply(d,bce_inst) - beta * tf.math.multiply(1-d, bce_inst))) 
    return loss