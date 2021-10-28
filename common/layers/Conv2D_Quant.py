import tensorflow as tf

class Conv2D_Quant(tf.keras.layers.Conv2D):
    # Doesnt really work well, and by not well I mean not at all
    def call(self, inputs):
        # Loss kernel
        kernel = self.kernel
        intValuesKernel = tf.clip_by_value(tf.math.round(kernel), -128, 127)
        absLossKernel = tf.reduce_sum(tf.math.square(kernel - intValuesKernel))
        # Loss bias
        bias = self.bias
        intValuesBias = tf.clip_by_value(tf.math.round(bias), -2147483648, 2147483647)
        absLossBias = tf.reduce_sum(tf.math.square(bias - intValuesBias))
        # Normalize to number of weights
        numKernel = tf.dtypes.cast(tf.size(kernel), tf.float32)
        numBias = tf.dtypes.cast(tf.size(bias), tf.float32)
        normalizedLoss = (absLossKernel + absLossBias) / ((numKernel + numBias) * 0.25 * 1.5)
        self.add_loss(normalizedLoss)

        return super().call(inputs)
