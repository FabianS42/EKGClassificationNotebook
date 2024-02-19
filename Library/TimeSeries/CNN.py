import tensorflow as tf

#arxiv 180500794
class ConvResidual(tf.keras.layers.Layer):
  def __init__(self, filters, kernal_size):
    super(ConvResidual, self).__init__()

    self.Layer1 = tf.keras.layers.Convolution1D(filters,kernal_size, padding='same')
    self.Relu1 = tf.keras.layers.ReLU()
    self.Layer2 = tf.keras.layers.Convolution1D(filters,kernal_size, padding='same')
    self.Relu2 = tf.keras.layers.ReLU()
    self.Pool = tf.keras.layers.MaxPool1D(kernal_size,2, padding='same')

  def call(self, x):

    y = self.Layer1(x)
    y = self.Relu1(y)
    y = self.Layer2(y)
   
    y = y+x

    y = self.Relu2(y)
    y = self.Pool(y)

    return y



class Conv(tf.keras.Model):
  def __init__(self, filters, kernal_size, output_size, num_residual_layers):
    super(Conv, self).__init__()

    self.Layer1 = tf.keras.layers.Convolution1D(filters,kernal_size, strides = 1, padding='same',name="input")
    self.ConvLayers = [ ConvResidual(filters, kernal_size) for _ in range(num_residual_layers)]
    self.Layer2 = tf.keras.layers.Dense(output_size)
    self.flatten = tf.keras.layers.Flatten()
    self.num_residual_layers = num_residual_layers

  def call(self, x):

    print("input:",x.shape)
    y = self.Layer1(x)
    print("after 1 layer:",y.shape)
    for i in range(self.num_residual_layers):
      y = self.ConvLayers[i](y)
      print("residual layer:",y.shape)

    y = self.flatten(y)
    print("flatten",y.shape)
    y = self.Layer2(y)
    print("out",y.shape)

    return y
  




class ConvResidual(tf.keras.layers.Layer):
  def __init__(self):
    super(ConvResidual, self).__init__()

    self.Layer1 = tf.keras.layers.Convolution1D(32,5, padding='same')
    self.Relu1 = tf.keras.layers.ReLU()
    self.Layer2 = tf.keras.layers.Convolution1D(32,5, padding='same')
    self.Relu2 = tf.keras.layers.ReLU()
    self.Pool = tf.keras.layers.MaxPool1D(5,2, padding='same')

  def call(self, x):

    y = self.Layer1(x)
    y = self.Relu1(y)
    y = self.Layer2(y)
   
    y = y+x

    y = self.Relu2(y)
    y = self.Pool(y)

    return y

class ECG_CNN_Arxiv_180500794(tf.keras.Model):
  def __init__(self):
    super(ECG_CNN_Arxiv_180500794, self).__init__()

    self.Layer1 = tf.keras.layers.Convolution1D(32,5, padding='same')

    self.ConvLayers = [
      ConvResidual()
    for _ in range(5)]

    self.Layer2 = tf.keras.layers.Dense(32)
    self.Relu = tf.keras.layers.ReLU() 
    self.Layer3 = tf.keras.layers.Dense(5, activation='softmax')
    self.flatten = tf.keras.layers.Flatten()

  def call(self, x):

    #print(x.shape)
    y = self.Layer1(x)
    #print(y.shape)
    
    for i in range(5):
      y = self.ConvLayers[i](y)
      #print(y.shape)

    y = self.Layer2(y)
    y = self.Relu(y)
    #print(y.shape)
    
    y = self.flatten(y)
    y = self.Layer3(y)
    #print(y.shape)

    return y