import numpy                as np
import tensorflow           as tf 
import pandas               as pd 
import matplotlib.pyplot    as plt

from   typing               import Tuple

from utils import *


def _wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)


class TadGan(tf.keras.Model):

    def __init__(self,
                 signal_input_shape: Tuple[int] = (750, 3),
                 latent_dim: int = 150,
                 gradient_penalty_weight: int = 10,
                 critic_iteration: int = 5,
                 batch_size = 32,
                
                 dropout = 0.2,
                 encoder_lstm_units: int = 100,
                 decoder_lstm_units: int = 100,
                 c_x_cnn_blocks: int = 4,
                 c_x_cnn_filters: int = 64,
                 c_x_cnn_kernel_size: int = 5,

                 print_summary: bool = True,
                 print_model_summary: bool = True,
                 log_all_losses: bool = True):
        
        super(TadGan, self).__init__()
        
        self.signal_input_shape = signal_input_shape
        self.signal_lenght = signal_input_shape[0]       
        self.n_signal_dim = signal_input_shape[1]
        self.latent_dim = latent_dim
        self.latent_shape = (self.latent_dim, 1)
        self.dropout = dropout

        self.gradient_penalty_weight = gradient_penalty_weight

        self.critic_iteration = critic_iteration
        self.encoder_lstm_units = encoder_lstm_units
        self.decoder_lstm_units = decoder_lstm_units
        self.c_x_cnn_blocks = c_x_cnn_blocks
        self.c_x_cnn_filters = c_x_cnn_filters
        self.c_x_cnn_kernel_size = c_x_cnn_kernel_size

        self.batch_size = batch_size

        self.encoder = self.build_encoder(x_in_shape=self.signal_input_shape, x_out_dim=self.latent_dim, lstm_units=self.encoder_lstm_units)
        self.decoder = self.build_decoder(x_in_shape=self.latent_shape, x_out_shape=self.signal_input_shape, lstm_units=self.decoder_lstm_units)

        self.critic_x = self.build_critic_x(x_in_shape=self.signal_input_shape, cnn_blocks=self.c_x_cnn_blocks,
                                             cnn_filters=self.c_x_cnn_filters, kernel_s=self.c_x_cnn_kernel_size,
                                               dropout=self.dropout)
        
        self.critic_z = self.build_critic_z(x_in_shape=self.latent_shape, dropout=self.dropout)

        self.log_all_losses = log_all_losses

        if print_summary:
            to_print_names = ['Input shape',
                              'Latent shape',
                              'Gradient penalty',
                              'Critic iterations',
                              'Encoder LSTM units',
                              'Decoder LSTM units',
                              'Critic X CNN blocks',
                              'Critic X CNN filters',
                              'Dropout']
            
            to_print_values = [self.signal_input_shape,
                               self.latent_shape,
                               self.gradient_penalty_weight,
                               self.critic_iteration,
                               self.encoder_lstm_units,
                               self.decoder_lstm_units,
                               self.c_x_cnn_blocks,
                               self.c_x_cnn_filters,
                               self.dropout]
            
            print_table(to_print_names, to_print_values)

        
            
        if print_model_summary:
            print('######################')
            print(self.encoder.summary())
            print('######################')
            print(self.decoder.summary())
            print('######################')
            print(self.critic_x.summary())
            print('######################')
            print(self.critic_z.summary())
            print('######################')

        self.build(input_shape=(None, signal_input_shape[0], signal_input_shape[1]))

    def compile(self,
                 encoder_decoder_opt,
                 c_x_opt,
                 c_z_opt,
                 encoder_decoder_loss,
                 c_x_loss,
                 c_z_loss,
                 **kwargs):

        super(TadGan, self).compile(**kwargs)
        
        self.encoder_decoder_opt = encoder_decoder_opt
        self.critic_x_opt = c_x_opt
        self.critic_z_opt = c_z_opt
        self.encoder_decoder_loss = encoder_decoder_loss
        self.critic_x_loss = c_x_loss
        self.critic_z_loss = c_z_loss

    @tf.function
    def call(self, inputs):

        out_enc = self.encoder(inputs)
        out_dec = self.decoder(out_enc) 
        out_c_x = self.critic_x(inputs)
        out_c_z = self.critic_z(out_enc)

        return out_dec, out_enc, out_c_x, out_c_z

    def build_encoder(self, x_in_shape, x_out_dim, lstm_units):

        x_in = tf.keras.layers.Input(shape=x_in_shape)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units, return_sequences=True))(x_in)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(x_out_dim)(x)
        x = tf.keras.layers.Reshape(target_shape=(x_out_dim, 1))(x)

        model = tf.keras.Model(x_in, x, name='Encoder')
        
        return model
    

    def build_decoder(self, x_in_shape, x_out_shape, lstm_units):

        x_in = tf.keras.layers.Input(shape=x_in_shape)

        x = tf.keras.layers.Flatten()(x_in)

        '''NOTA:
            si potrebbe ricostruire metà segnale subito e l'altra metà in mezzo agli LSTM. 
            Problema: 
                Serve segnale di lunghezza pari?
            Vantaggio:
                Performance migliori su segnali complessi
            Per ora lo costruisco così perchè il train è fatto su 750 accelerazioni'''
        
        half_signal = self.signal_lenght // 2

        x = tf.keras.layers.Dense(half_signal)(x)
        x = tf.keras.layers.Reshape(target_shape=(half_signal, 1))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units, return_sequences=True))(x)
        x = tf.keras.layers.UpSampling1D(size=2)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units, return_sequences=True))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_signal_dim))(x)
        x = tf.keras.layers.Activation('tanh')(x)

        model = tf.keras.Model(x_in, x, name='Decoder')
        
        return model
    
    def build_critic_x(self, x_in_shape, cnn_blocks, cnn_filters, kernel_s, dropout):

        x_in = tf.keras.layers.Input(shape=x_in_shape)

        x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=kernel_s)(x_in)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        if cnn_blocks > 1:
            for i in range(cnn_blocks-1):
                x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=kernel_s)(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
                x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(x_in, x, name='Critic_X')

        return model


    def build_critic_z(self, x_in_shape, dropout):

        x_in = tf.keras.layers.Input(shape=x_in_shape)

        x = tf.keras.layers.Flatten()(x_in)
        
        x = tf.keras.layers.Dense(x_in_shape[0])(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(dropout)(x)        
        
        x = tf.keras.layers.Dense(x_in_shape[0])(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(x_in, x, name='Critic_Z')

        return model
    
    @tf.function 
    def c_x_gradient_penalty(self, batch_size, true, pred):

        r = tf.keras.backend.random_uniform((batch_size, 1, 1))

        i = (r * true) + ((1 - r) * pred)
        with tf.GradientTape() as gt:

            gt.watch(i)
            pred_i = self.critic_x(i)

        gradient = gt.gradient(pred_i, [i])[0]
        g_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2]))
        gp = tf.reduce_mean((g_norm - 1.0) ** 2)

        return gp


    @tf.function 
    def c_z_gradient_penalty(self, critic_batch_size, true, pred):

        r = tf.keras.backend.random_uniform((critic_batch_size, 1, 1))
        i = (r * true) + ((1 - r) * pred)

        with tf.GradientTape() as gt:

            gt.watch(i)
            pred_i = self.critic_z(i)
        
        gradient = gt.gradient(pred_i, [i])[0]
        g_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2]))
        gp = tf.reduce_mean((g_norm - 1.0) ** 2)

        return gp
    

    @tf.function
    def compute_critic_x_loss(self, x, z, critic_batch_size):

        fake_x = self.decoder(z)
        valid_x = self.critic_x(x)
        not_valid_x = self.critic_x(fake_x)

        true_tensor = -tf.ones_like(valid_x)
        false_tensor = tf.ones_like(not_valid_x)

        loss_1 = self.critic_x_loss(y_true=true_tensor, y_pred=valid_x)
        loss_2 = self.critic_x_loss(y_true=false_tensor, y_pred=not_valid_x)
        gp_loss = self.c_x_gradient_penalty(critic_batch_size, x, fake_x)

        loss = loss_1 + loss_2 + self.gradient_penalty_weight * gp_loss

        return loss, loss_1, loss_2, gp_loss
    

    @tf.function
    def compute_critic_z_loss(self, x, z, critic_batch_size):

        fake_z = self.encoder(x)

        valid_z = self.critic_z(z)
        not_valid_z = self.critic_z(fake_z)

        true_tensor = -tf.ones_like(valid_z)
        false_tensor = tf.ones_like(not_valid_z)

        loss_1 = self.critic_z_loss(y_true=true_tensor, y_pred=valid_z)
        loss_2 = self.critic_z_loss(y_true=false_tensor, y_pred=not_valid_z)
        gp_loss = self.c_z_gradient_penalty(critic_batch_size, z, fake_z)

        loss = loss_1 + loss_2 + self.gradient_penalty_weight * gp_loss

        return loss, loss_1, loss_2, gp_loss


    @tf.function 
    def compute_enc_dec_loss(self, x, z):

        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        fake_x = self.decoder(z)

        critic_z_fake_z = self.critic_z(encoded_x)
        critic_x_fake_x = self.critic_x(fake_x)

        encoder_loss_z = self.encoder_decoder_loss(y_pred=critic_z_fake_z, y_true=-tf.ones_like(critic_z_fake_z))
        decoder_loss_x = self.encoder_decoder_loss(y_pred=critic_z_fake_z, y_true=-tf.ones_like(critic_z_fake_z))

        general_reconstruction_cost = tf.reduce_mean(tf.square((x - decoded_x)))

        loss = encoder_loss_z + decoder_loss_x + 10 * general_reconstruction_cost
    
        return loss, encoder_loss_z, decoder_loss_x, general_reconstruction_cost
    

    @tf.function 
    def train_step(self, X):

        if isinstance(X, tuple):
            X = X[0]
        
        batch_size = self.batch_size
        mini_batch_size = batch_size // self.critic_iteration

        c_x_loss_ = []
        c_z_loss_ = []

        for k in range(self.critic_iteration):
            
            z = tf.random.normal(shape=(mini_batch_size, self.latent_dim, 1), mean=0.0, stddev=1, dtype=tf.dtypes.float32, seed=42)
            x_mb = X[k * mini_batch_size: (k + 1) * mini_batch_size]

            with tf.GradientTape() as gt_x:
                critic_x_loss = self.compute_critic_x_loss(x_mb, z, mini_batch_size)

            c_x_gradient = gt_x.gradient(critic_x_loss[0], self.critic_x.trainable_variables)
            self.critic_x_opt.apply_gradients(zip(c_x_gradient, self.critic_x.trainable_variables))

            c_x_loss_.append([i for i in critic_x_loss])

            with tf.GradientTape() as gt_z:
                critic_z_loss = self.compute_critic_z_loss(x_mb, z, mini_batch_size)
            
            c_z_gradient = gt_z.gradient(critic_z_loss[0], self.critic_z.trainable_variables)
            self.critic_z_opt.apply_gradients(zip(c_z_gradient, self.critic_z.trainable_variables))

            c_z_loss_.append([i for i in critic_z_loss])
        
        with tf.GradientTape() as gt_e_d:
          # Dovrei mettere x_mb????
            e_d_loss = self.compute_enc_dec_loss(X, z)

        e_d_gradient = gt_e_d.gradient(e_d_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.encoder_decoder_opt.apply_gradients(zip(e_d_gradient, self.encoder.trainable_variables + self.decoder.trainable_variables))

        c_x_loss_tot = tf.reduce_mean(tf.convert_to_tensor(c_x_loss_), axis=1)
        # c_x_loss_tot = np.mean(np.array(c_x_loss_))
        c_z_loss_tot =  tf.reduce_mean(tf.convert_to_tensor(c_z_loss_), axis=1)
        enc_gen_loss_tot = e_d_loss

        if self.log_all_losses:
            loss_dict = {
                "Cx_total": c_x_loss_tot[0],
                "Cx_valid": c_x_loss_tot[1],
                "Cx_fake": c_x_loss_tot[2],
                "Cx_gp_penalty": c_x_loss_tot[3],

                "Cz_total": c_z_loss_tot[0],
                "Cz_valid": c_z_loss_tot[1],
                "Cz_fake": c_z_loss_tot[2],
                "Cz_gp_penalty": c_z_loss_tot[3],

                "EG_total": enc_gen_loss_tot[0],
                "EG_fake_gen_x": enc_gen_loss_tot[1],
                "EG_fake_gen_z": enc_gen_loss_tot[2],
                "G_rec": enc_gen_loss_tot[3],
            }
        else:
            loss_dict = {
                "Cx_total": c_x_loss_tot[0],
                "Cz_total": c_z_loss_tot[0],
                "EG_total": enc_gen_loss_tot[0]
            }

        return loss_dict


    @tf.function
    def test_step(self, X):

        if isinstance(X, tuple):
            X = X[0]

        batch_size = self.batch_size

        z = tf.random.normal(shape=(batch_size, self.latent_dim, 1))

        c_x_loss_tot = self.compute_critic_x_loss(X, z, batch_size)
        c_z_loss_tot = self.compute_critic_z_loss(X, z, batch_size)
        enc_gen_loss_tot = self.compute_enc_dec_loss(X, z)

        if self.log_all_losses:
            loss_dict = {
                "Cx_total": c_x_loss_tot[0],
                "Cx_valid": c_x_loss_tot[1],
                "Cx_fake": c_x_loss_tot[2],
                "Cx_gp_penalty": c_x_loss_tot[3],

                "Cz_total": c_z_loss_tot[0],
                "Cz_valid": c_z_loss_tot[1],
                "Cz_fake": c_z_loss_tot[2],
                "Cz_gp_penalty": c_z_loss_tot[3],

                "EG_total": enc_gen_loss_tot[0],
                "EG_fake_gen_x": enc_gen_loss_tot[1],
                "EG_fake_gen_z": enc_gen_loss_tot[2],
                "G_rec": enc_gen_loss_tot[3],
            }
        else:
            loss_dict = {
                "Cx_total": c_x_loss_tot[0],
                "Cz_total": c_z_loss_tot[0],
                "EG_total": enc_gen_loss_tot[0]
            }

        return loss_dict


    def fit(self,
            x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):

        if not isinstance(validation_data, tf.data.Dataset):
            if (validation_data is not None) and (validation_batch_size is None):
                validation_batch_size = batch_size

        if not isinstance(x, tf.data.Dataset):
            batch_size = batch_size * self.n_iterations_critic

        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle,
                           class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                           validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
    

    @staticmethod
    def export_as_keras_model(tadgan_model, export_path: str):
        x = tf.keras.layers.Input(shape=tadgan_model.ts_input_shape, name="ts_input")
        latent_encoding = tadgan_model.encoder(x)
        y_hat = tadgan_model.generator(latent_encoding)
        critic_x = tadgan_model.critic_x(x)
        critic_z = tadgan_model.critic_z(latent_encoding)

        standalone_model = tf.keras.models.Model(inputs=x, outputs=[y_hat, latent_encoding, critic_x, critic_z])
        standalone_model.save(export_path)