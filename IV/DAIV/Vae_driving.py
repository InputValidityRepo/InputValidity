'''
Code is implemented over the baseline provided in keras git repo
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate SVHN inputs by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from data_utils import load_train_data, load_test_data
import numpy as np

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def Vae_driving(input_tensor=None, train=False, weighted_kl=False):
    #input img size (104,104) due to Conv2DTranspose shape does not match
    np.random.seed(0)
    if train:
        # network parameters
        input_shape = (image_size, image_size, image_chn)
        print('input shape', input_shape)
        input_tensor = Input(shape=input_shape)

        batch_size = 256
        epochs = 200

    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()

    latent_dim = 512
    
    # VAE model = encoder + decoder
    # build encoder model
    x = Conv2D(8, (5, 5), padding='same', strides=(2, 2), activation='relu')(input_tensor)
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    #x = Conv2D(128, (5, 5), padding='same', activation='relu')(x)
    #x = Conv2D(128, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
      
    
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(input_tensor, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    #x = Conv2DTranspose(128, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    #x = Conv2DTranspose(128, (5, 5), padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = Conv2DTranspose(64, (5, 5), padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = Conv2DTranspose(16, (5, 5), padding='same', activation='relu')(x)
    x = Conv2DTranspose(8, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    pos_mean = Conv2DTranspose(3, (5, 5), padding='same', name='pos_mean')(x)
    pos_log_var = Conv2DTranspose(3, (5, 5), padding='same', name='pos_log_var')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, [pos_mean, pos_log_var], name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(input_tensor)[2])
    vae = Model(input_tensor, outputs, name='vae_driving')
    vae.summary()
    
    if train:
        # VAE loss = reconstruction_loss + kl_loss
        loss_a = float(np.log(2 * np.pi)) + outputs[1]
        loss_m = K.square(outputs[0] - input_tensor) / K.exp(outputs[1])
        reconstruction_loss = -0.5 * K.sum((loss_a + loss_m), axis=[-1,-2, -3])

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(-reconstruction_loss + 0.1 * kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer="adam")
        vae.summary()
        model_checkpoint_callback = ModelCheckpoint(
            filepath='./vae_driving_01.h5',
            save_weights_only=True,
            save_best_only=False,
            period=10)

        vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None), verbose=2, callbacks=[model_checkpoint_callback])
        # save model
        # vae.save_weights('./vae_driving.h5')
    else:
        if weighted_kl:
            print('load weights vae_driving_01.h5')
            vae.load_weights('./vae_driving_01.h5')
        else:
            print('load weights vae_driving.h5')
            vae.load_weights('./vae_driving.h5')

    return vae

if __name__ == '__main__':
    image_size = 104
    image_chn = 3
    train_data, test_data = load_train_data(), load_test_data()
    x_train = np.reshape(train_data, [-1, image_size, image_size, image_chn])
    x_test = np.reshape(test_data, [-1, image_size, image_size, image_chn])
    vae = Vae_driving(train=True)
