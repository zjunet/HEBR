# -*- coding: utf-8 -*-


import tensorflow as tf


class HEBR_Cell():
    def __init__(self):
        self.dims = {'user': None,
                     'area': None,
                     'climate': None,
                     'user_hidden': None,
                     'area_hidden': None,
                     'climate_hidden': None,
                     'user_area': None,
                     'user_climate': None,
                     'user_area_hidden': None,
                     'user_climate_hidden': None,
                     'user_area_climate': None,
                     'user_area_climate_hidden': None}
        self.is_training = True

    def set_configuration(self, his_len, dims, is_training):
        self.his_len = his_len
        self.dims = dims
        self.is_training = is_training

    def __weights__(self, input_dim, output_dim, name, init=True, std=0.1, reg=None):
        if init:
            return tf.get_variable(name, shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(0.0, std), regularizer=reg)
        else:
            return tf.get_variable(name, shape=[input_dim, output_dim])

    def __bias__(self, output_dim, name, init=True):
        if init:
            return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))
        else:
            return tf.get_variable(name, shape=[output_dim])


    def FeatureFusion(self, emb1, emb2, prev_emb2, dims, name, reuse=False):
        """
        Multi-step hierarchical fusion mechanism
        """
        dim_1 = dims[0]
        dim_2 = dims[1]
        with tf.variable_scope(name, reuse=reuse):
            self.W_cur = self.__weights__(dim_2, dim_1, name='Cur_weight', init=self.is_training)
            self.W_prev = self.__weights__(dim_2, dim_1, name='Prev_weight', init=self.is_training)
            self.Wa = self.__weights__(2*dim_1, 1, name='Attention_weight', init=self.is_training)

            f_cur = (emb1 + tf.matmul(emb2, self.W_cur)) / 2
            f_prev = (emb1 + tf.matmul(prev_emb2, self.W_prev)) / 2
            fused_emb = tf.concat([f_cur, f_prev], axis=-1)

            a_score = tf.matmul(fused_emb, self.Wa)
            out_logits = a_score * tf.nn.tanh(fused_emb)

        return out_logits, fused_emb, a_score

    def __LSTMUnit__(self, input_x, prev, prev_memory, dims, name, reuse=False):
        """
        LSTM unit
        """
        input_dim = dims[0]
        output_dim = dims[1]

        with tf.variable_scope(name, reuse=reuse):
            self.Wi = self.__weights__(input_dim, output_dim, name='Input_weight_1', init=self.is_training)
            self.Ui = self.__weights__(output_dim, output_dim, name='Input_weight_2', init=self.is_training)
            self.bi = self.__bias__(output_dim, name='Input_bias', init=self.is_training)

            self.Wf = self.__weights__(input_dim, output_dim, name='Forget_weight_1', init=self.is_training)
            self.Uf = self.__weights__(output_dim, output_dim, name='Forget_weight_2', init=self.is_training)
            self.bf = self.__bias__(output_dim, name='Forget_bias', init=self.is_training)

            self.Wo = self.__weights__(input_dim, output_dim, name='Output_weight_1', init=self.is_training)
            self.Uo = self.__weights__(output_dim, output_dim, name='Output_weight_2', init=self.is_training)
            self.bo = self.__bias__(output_dim, name='Output_bias', init=self.is_training)

            self.Wc = self.__weights__(input_dim, output_dim, name='Global_weight_1', init=self.is_training)
            self.Uc = self.__weights__(output_dim, output_dim, name='Global_weight_2', init=self.is_training)
            self.bc = self.__bias__(output_dim, name='Global_bias', init=self.is_training)

            # input gate
            I = tf.nn.sigmoid(tf.matmul(input_x, self.Wi) + tf.matmul(prev, self.Ui) + self.bf)
            # forget gate
            F = tf.nn.sigmoid(tf.matmul(input_x, self.Wf) + tf.matmul(prev, self.Uf) + self.bf)
            # output gate
            O = tf.nn.sigmoid(tf.matmul(input_x, self.Wo) + tf.matmul(prev, self.Uo) + self.bo)
            # long term memory cell
            C_ = tf.nn.tanh(tf.matmul(input_x, self.Wc) + tf.matmul(F * prev, self.Uc) + self.bc)
            # output
            Ct = F * prev_memory + I * C_
            # current information
            current_memory = Ct
            current = O * tf.nn.tanh(Ct)

        return current, current_memory

    def Cell(self, input_u, input_a, input_c, hiddens, reuse=False):

        prev_emb_u = hiddens['uh']
        prev_mem_u = hiddens['um']
        prev_emb_a = hiddens['ah']
        prev_mem_a = hiddens['am']
        prev_emb_c = hiddens['ch']
        prev_mem_c = hiddens['cm']
        prev_emb_ua = hiddens['uah']
        prev_mem_ua = hiddens['uam']
        prev_emb_uc = hiddens['uch']
        prev_mem_uc = hiddens['ucm']
        prev_emb_uac = hiddens['uach']
        prev_mem_uac = hiddens['uacm']

        with tf.variable_scope('HEBR_Cell'):
            cur_emb_u, cur_mem_u = self.__LSTMUnit__(input_u, prev_emb_u, prev_mem_u, dims=[self.dims['user'], self.dims['user_hidden']], name='User', reuse=reuse)
            cur_emb_a, cur_mem_a = self.__LSTMUnit__(input_a, prev_emb_a, prev_mem_a, dims=[self.dims['area'], self.dims['area_hidden']], name='Area', reuse=reuse)
            cur_emb_c, cur_mem_c = self.__LSTMUnit__(input_c, prev_emb_c, prev_mem_c, dims=[self.dims['climate'], self.dims['climate_hidden']], name='Climate', reuse=reuse)

            input_ua, fuse_ua, score_ua = self.FeatureFusion(cur_emb_u, cur_emb_a, prev_emb_a, dims=[self.dims['user_hidden'], self.dims['area_hidden']], name='User_Area_Fusion', reuse=reuse)
            input_uc, fuse_uc, score_uc = self.FeatureFusion(cur_emb_u, cur_emb_c, prev_emb_c, dims=[self.dims['user_hidden'], self.dims['climate_hidden']], name='User_Climate_Fusion', reuse=reuse)

            cur_emb_ua, cur_mem_ua = self.__LSTMUnit__(input_ua, prev_emb_ua, prev_mem_ua, dims=[self.dims['user_area'], self.dims['user_area_hidden']], name='User_Area', reuse=reuse)
            cur_emb_uc, cur_mem_uc = self.__LSTMUnit__(input_uc, prev_emb_uc, prev_mem_uc, dims=[self.dims['user_climate'], self.dims['user_climate_hidden']], name='User_Climate', reuse=reuse)

            input_uac, fuse_uac, score_uac = self.FeatureFusion(cur_emb_ua, cur_emb_uc, prev_emb_uc, dims=[self.dims['user_area_hidden'], self.dims['user_climate_hidden']], name='User_Area_Climate_Fusion', reuse=reuse)

            cur_emb_uac, cur_mem_uac = self.__LSTMUnit__(input_uac, prev_emb_uac, prev_mem_uac, dims=[self.dims['user_area_climate'], self.dims['user_area_climate_hidden']], name='User_Area_Climate', reuse=reuse)

            return {'uh': cur_emb_u,
                    'um': cur_mem_u,
                    'ah': cur_emb_a,
                    'am': cur_mem_a,
                    'ch': cur_emb_c,
                    'cm': cur_mem_c,
                    'uah': cur_emb_ua,
                    'uam': cur_mem_ua,
                    'uch': cur_emb_uc,
                    'ucm': cur_mem_uc,
                    'uach': cur_emb_uac,
                    'uacm': cur_mem_uac,
                    'uas': score_ua,
                    'ucs': score_uc,
                    'uacs': score_uac}


    def get_embedding(self, user_seq, area_seq, climate_seq):
        self.batch_size = tf.shape(user_seq)[0]

        # time major
        user_seq = tf.transpose(user_seq, [1, 0, 2])  # [seq_length * batch_size * n_features]
        area_seq = tf.transpose(area_seq, [1, 0, 2])
        climate_seq = tf.transpose(climate_seq, [1, 0, 2])


        # inital
        hiddens = {'uh': tf.zeros([self.batch_size, self.dims['user_hidden']], dtype=tf.float32),
                    'um': tf.zeros([self.batch_size, self.dims['user_hidden']], dtype=tf.float32),
                    'ah': tf.zeros([self.batch_size, self.dims['area_hidden']], dtype=tf.float32),
                    'am': tf.zeros([self.batch_size, self.dims['area_hidden']], dtype=tf.float32),
                    'ch': tf.zeros([self.batch_size, self.dims['climate_hidden']], dtype=tf.float32),
                    'cm': tf.zeros([self.batch_size, self.dims['climate_hidden']], dtype=tf.float32),
                    'uah': tf.zeros([self.batch_size, self.dims['user_area_hidden']], dtype=tf.float32),
                    'uam': tf.zeros([self.batch_size, self.dims['user_area_hidden']], dtype=tf.float32),
                    'uch': tf.zeros([self.batch_size, self.dims['user_climate_hidden']], dtype=tf.float32),
                    'ucm': tf.zeros([self.batch_size, self.dims['user_climate_hidden']], dtype=tf.float32),
                    'uach': tf.zeros([self.batch_size, self.dims['user_area_climate_hidden']], dtype=tf.float32),
                    'uacm': tf.zeros([self.batch_size, self.dims['user_area_climate_hidden']], dtype=tf.float32),
                    'uas': None,
                    'ucs': None,
                    'uacs': None}

        emb_logits, attention_logits = [], []


        for i in range(self.his_len):
            if i == 0:
                hiddens = self.Cell(user_seq[i], area_seq[i], climate_seq[i], hiddens, reuse=False)
            else:
                hiddens = self.Cell(user_seq[i], area_seq[i], climate_seq[i], hiddens, reuse=True)
            emb_logits.append(hiddens['uach'])
            attention_logits.append([hiddens['uas'], hiddens['ucs'], hiddens['uacs']])

        return emb_logits, attention_logits