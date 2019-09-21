import datetime
import json
import os
import re
import string
import unicodedata

import numpy as np
import tensorflow as tf

from model.base_model import BaseModel
from model.utils import (build_gru_cell, build_gru_cell_with_dropout,
                         build_lstm_layer_norm,
                         stack_bidirectional_dynamic_rnn,
                         stack_bidirectional_dynamic_rnn_cnn,
                         get_logger)

s1 = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"
letters = list(set(s1.lower()))


def data_generator(all_data, batch_size, shuffle=True):
    if shuffle:
        np.random.shuffle(all_data)
    i = 0
    while i < len(all_data):
        if i + batch_size > len(all_data):
            yield all_data[i:]
        else:
            yield all_data[i:i+batch_size]
        i += batch_size


class Tokenizer(object):
    def __init__(self, file, oov_token='<unk>', pad_val=0):
        with open(file, 'r') as _in:
            self.dictionary = {x: int(i) for x, i in map(
                lambda x: x.strip().split('\t'), _in.readlines())}
        self.oov_token = oov_token
        self.pad_val = pad_val
        self.idx2word = {i: x for x, i in self.dictionary.items()}

    def noise_maker(self, sentence, threshold):
        noisy_sentence = []
        i = 0
        while i < len(sentence):
            random = np.random.uniform(0, 1, 1)
            if random < threshold:
                noisy_sentence.append(sentence[i])
            else:
                new_random = np.random.uniform(0, 1, 1)
                if new_random > 0.67:
                    if i == (len(sentence) - 1):
                        continue
                    else:
                        noisy_sentence.append(sentence[i+1])
                        noisy_sentence.append(sentence[i])
                        i += 1
                elif new_random < 0.33:
                    random_letter = np.random.choice(letters, 1)[0]
                    noisy_sentence.append(random_letter)
                    noisy_sentence.append(sentence[i])
                else:
                    pass
            i += 1
        return ''.join(noisy_sentence)

    def matrix2texts(self, matrix):
        return [
            ' '.join(''.join(self.idx2word[x] for x in word if x > 0) for word in line).strip() for line in matrix
        ]

    def _clean_text(self, text):
        text = text.strip()
        text = re.sub("\u2013|\u2014", "-", text)
        text = re.sub("\u00D7", " x ", text)
        text = unicodedata.normalize('NFKC', text)
        for i in string.punctuation:
            text = text.replace(i, ' {} '.format(i))
        text = re.sub(r'[^{}\s\w]+'.format(string.punctuation), ' ', text)
        text = re.sub(r'\d+', ' 0 ', text)
        return re.sub(r'\s+', ' ', text).lower()

    def text2array(self, text, threshold):
        text = self._clean_text(text)
        words = text.split()
        res = []
        noise_res = []
        word_len = []
        noise_words = [self.noise_maker(x, threshold) for x in words]
        noise_word_len = []
        # print(noise_words)
        for word, noise_word in zip(words, noise_words):
            word_len.append(len(word))
            noise_word_len.append(len(noise_word))
            res.append([
                self.dictionary.get(x, self.dictionary[self.oov_token]) for x in list(word)
            ])
            noise_res.append([
                self.dictionary.get(x, self.dictionary[self.oov_token]) for x in list(noise_word)
            ])
        return res, word_len, noise_res, noise_word_len

    def texts2matrix(self, texts, threshold=1.0):
        arrays = []
        word_len = []
        seq_len = []
        arrays_noise = []
        word_len_noise = []
        for text in texts:
            arr, wl, arr_n, wln = self.text2array(text, threshold)
            arrays.append(arr)
            word_len.append(wl)
            seq_len.append(len(wl))
            arrays_noise.append(arr_n)
            word_len_noise.append(wln)
        seq_len = np.array(seq_len, dtype=np.int32)
        padded_word_len = np.zeros(
            [len(word_len), seq_len.max()], dtype=np.int32)
        padded_word_len_noise = np.zeros(
            [len(word_len_noise), seq_len.max()], dtype=np.int32
        )
        for i in range(len(word_len)):
            padded_word_len[i][:len(word_len[i])] = word_len[i]
            padded_word_len_noise[i][:len(
                word_len_noise[i])] = word_len_noise[i]
        max_word_len = np.max(padded_word_len)
        max_word_len_noise = np.max(padded_word_len_noise)
        padded_arrays = np.zeros(
            [len(arrays), seq_len.max(), max_word_len], dtype=np.int32)
        padded_arrays_noise = np.zeros(
            [len(arrays_noise), seq_len.max(), max_word_len_noise], dtype=np.int32)
        for i in range(len(arrays)):
            for j in range(len(arrays[i])):
                padded_arrays[i][j][:len(arrays[i][j])] = arrays[i][j]
                padded_arrays_noise[i][j][:len(
                    arrays_noise[i][j])] = arrays_noise[i][j]
        return {
            'char_ids': padded_arrays_noise,
            'word_length': padded_word_len_noise,
            'sequence_length': seq_len,
            'target_char': padded_arrays,
            'target_char_len': padded_word_len
        }

    def num_chars(self):
        return len(self.dictionary)


class Config(object):
    def __init__(self):
        self.logger = None
        self.num_checkpoints = None
        self.checkpoint_prefix = None
        self.train_summary_dir = None
        self.encoder = None
        self.decoder = None
        self.concat_residual = False
        self.use_residual = False
        self.beam_width = 10
        self.length_penalty_weight = 0.0
        self.nchars = None
        self.cdims = None
        self.num_checkpoints = 10
        self.training_method = 'adam'
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.use_nesterov = True
        self.clip_grad = 'global'
        self.start_token_id = None
        self.end_token_id = 0
        self.tokenizer = None
        self.version = 1

    def set_log_dir(self, new_dir):
        self.log_dir = new_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.out_dir = os.path.abspath(os.path.join(
            self.log_dir, str(self.version)))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.saved_model_dir = os.path.abspath(
            os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.saved_model_dir, "model")
        self.train_summary_dir = os.path.join(
            self.out_dir, "summaries", "train")
        self.dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
        self.logger = get_logger(os.path.join(self.out_dir, 'log'))


class SpellingCorrector(BaseModel):
    def __init__(self, configs, mode='train'):
        super().__init__(configs)
        self.mode = mode

    def _add_placeholders(self):
        self.char_ids = tf.placeholder(
            tf.int32, [None, None, None], name='char_ids')
        self.sequence_length = tf.placeholder(
            tf.int32, [None],
            name='sequence_length'
        )
        self.word_length = tf.placeholder(
            tf.int32, [None, None],
            name='word_length'
        )
        self.start_token_id = tf.placeholder(
            tf.int32, [], name='start_token_id'
        )
        self.end_token_id = tf.placeholder(
            tf.int32, [], name='end_token_id'
        )
        self._s = tf.shape(self.char_ids)
        self.batch_size = self._s[0]*self._s[1]
        if self.mode == 'train':
            self.target_char = tf.placeholder(
                tf.int32, [None, None, None], name='char_ids'
            )
            s = tf.shape(self.target_char)
            self.target_char_len = tf.placeholder(
                tf.int32, [None, None], name='target_len')
            char_ids = tf.reshape(
                self.target_char, [self.batch_size, s[2]])
            self.target_char_input = tf.concat([
                tf.expand_dims(
                    tf.zeros([self.batch_size], dtype=tf.int32) +
                    self.start_token_id,
                    -1),
                char_ids
            ], -1)
            temp = tf.reshape(self.target_char_len, [-1])
            self.target_word_length = tf.add(
                temp,
                1,
                name='target_word_length'
            )
            self.target_char_output = tf.concat([
                char_ids,
                tf.zeros([self.batch_size, 1], dtype=tf.int32)
            ], -1)
            self.target_char_output += tf.one_hot(temp,
                                                  depth=tf.shape(self.target_char_output)[-1], dtype=tf.int32) * self.end_token_id
        self.loss = 0
        with tf.variable_scope('char_embedding'):
            self.char_embedding = tf.Variable(
                tf.random_uniform(
                    [self.configs.nchars, self.configs.cdims],
                    -0.1, 0.1,
                    tf.float32
                ),
                name='embedding_matrix'
            )
        if self.mode == 'train':
            self.dropout = tf.placeholder(tf.float32, [], name='dropout')
            # self.learning_rate = tf.placeholder(
            # tf.float32, [], name='learning_rate')
            self.rnn_cell = lambda hidden: build_gru_cell_with_dropout(
                hidden, self.dropout)
        else:
            self.dropout = None
            self.rnn_cell = build_gru_cell

    def _build_encoder(self):
        assert isinstance(self.configs.encoder, dict)
        with tf.variable_scope('encoder'):
            with tf.variable_scope('encoder_char_embedding'):
                char_embedding = tf.nn.embedding_lookup(
                    self.char_embedding, self.char_ids,
                    name='lookup'
                )
                s = tf.shape(char_embedding)
                char_embedding = tf.reshape(
                    char_embedding,
                    [s[0]*s[1], s[2], self.configs.cdims]
                )
                if self.configs.encoder['char_embedding'] == 'rnn':
                    with tf.variable_scope('char_rnn'):
                        word_length = tf.reshape(self.word_length, [s[0]*s[1]])
                        _, fs_fw, fs_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            [self.rnn_cell(x)
                             for x in self.configs.encoder['num_hidden_char']],
                            [self.rnn_cell(x)
                             for x in self.configs.encoder['num_hidden_char']],
                            char_embedding,
                            sequence_length=word_length,
                            dtype=tf.float32
                        )
                        output = tf.concat([fs_fw[-1], fs_bw[-1]], axis=-1)
                        output = tf.nn.relu(output)
                        final_size = self.configs.encoder['num_hidden_char'][-1] * 2
                else:
                    with tf.variable_scope('char_cnn'):
                        word_length = tf.reshape(
                            self.word_length, [self.batch_size])
                        mask = tf.expand_dims(tf.sequence_mask(word_length, dtype=tf.float32),
                                              axis=-1)
                        output = char_embedding
                        for kernel_size, num_filters in zip(self.configs.encoder['char_embedding_kernel_size'], self.configs.encoder['num_hidden_char']):
                            output *= tf.stop_gradient(tf.tile(mask,
                                                               multiples=[1, 1, output.get_shape()[-1]]))
                            output = tf.layers.conv1d(
                                inputs=output,
                                filters=num_filters,
                                kernel_size=kernel_size,
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer()
                            )
                        output = tf.reduce_max(output, axis=-2)
                        final_size = self.configs.encoder['num_hidden_char'][-1]
                output = tf.reshape(output, [s[0], s[1], final_size])
                encoder_word_emb = tf.nn.dropout(
                    x=output, keep_prob=self.dropout, noise_shape=[s[0], 1, output.get_shape()[2]]) if self.mode == 'train' else output
            mask = tf.expand_dims(tf.sequence_mask(self.sequence_length, dtype=tf.float32),
                                  axis=-1)
            if self.configs.encoder['type'] == 'cnn':
                with tf.variable_scope('encoder_cnn'):
                    output = encoder_word_emb
                    for ksz, filters, dl in zip(self.configs.encoder['kernel_size'], self.configs.encoder['num_filters'], self.configs.encoder['dilation_rate']):
                        output *= tf.stop_gradient(tf.tile(mask,
                                                           multiples=[1, 1, output.get_shape()[-1]]))
                        output = tf.layers.conv1d(
                            inputs=output,
                            filters=filters,
                            kernel_size=ksz,
                            dilation_rate=dl,
                            padding='same',
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer()
                        )
                        if self.configs.encoder['gate']:
                            gate = tf.layers.conv1d(
                                inputs=output,
                                filters=filters,
                                kernel_size=ksz,
                                dilation_rate=dl,
                                padding='same',
                                activation=tf.sigmoid,
                                kernel_initializer=tf.contrib.layers.xavier_initializer()
                            )
                            output *= gate
            else:
                with tf.variable_scope('encoder_rnn'):
                    output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        [self.rnn_cell(x)
                         for x in self.configs.encoder['num_hidden']],
                        [self.rnn_cell(x)
                         for x in self.configs.encoder['num_hidden']],
                        encoder_word_emb,
                        sequence_length=self.sequence_length,
                        dtype=tf.float32
                    )
            with tf.variable_scope('encoder_output'):
                if self.configs.encoder['concat']:
                    output = tf.concat([output, encoder_word_emb], -1)
                self.encoder_output = output * tf.stop_gradient(tf.tile(mask,
                                                                        multiples=[1, 1, output.get_shape()[-1]]))

    def _build_decoder(self):
        # assert self.encoder_output
        assert isinstance(self.configs.decoder, dict)
        with tf.variable_scope('decoder') as decoder_scope:
            if self.mode == 'train':
                with tf.variable_scope('decoder_char_embedding'):
                    decoder_emb_inp = tf.nn.embedding_lookup(
                        self.char_embedding, self.target_char_input,
                        name='lookup'
                    )
            num_unit = self.encoder_output.get_shape().as_list()[-1]
            cell = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell(num_unit)
                                                for _ in range(self.configs.decoder['num_layers'])])
            self.output_layer = tf.layers.Dense(
                self.configs.nchars, use_bias=False, name="output_projection")
            if self.mode == 'train':
                if self.configs.decoder['teacher']:
                    helper = tf.contrib.seq2seq.TrainingHelper(
                        decoder_emb_inp, self.target_word_length, time_major=False
                    )
                else:
                    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                        inputs=decoder_emb_inp, sequence_length=self.target_word_length,
                        embedding=self.char_embedding,
                        sampling_probability=self.configs.decoder['sampling'],
                        time_major=False)
                init_state = tf.reshape(self.encoder_output,
                                        [self.batch_size, self.encoder_output.get_shape()[-1]], 'decoder_initial_state')
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    (init_state,) * self.configs.decoder['num_layers']
                )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    output_time_major=False,
                    swap_memory=True,
                    scope=decoder_scope
                )
                self.sample_id_train = outputs.sample_id
                self.logits_train = self.output_layer(outputs.rnn_output)
            # if self.configs.decoder['mode'] == 'beam':
            start_tokens = tf.fill([self.batch_size], self.start_token_id)
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                tf.reshape(self.encoder_output,
                           [self.batch_size, self.encoder_output.get_shape()[-1]]), multiplier=self.configs.beam_width, name='decoder_initial_state')
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cell,
                embedding=self.char_embedding,
                start_tokens=start_tokens,
                end_token=self.end_token_id,
                initial_state=(decoder_initial_state,) *
                self.configs.decoder['num_layers'],
                beam_width=self.configs.beam_width,
                output_layer=self.output_layer,
                length_penalty_weight=self.configs.length_penalty_weight)
            # else:

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=False,
                swap_memory=True,
                scope=decoder_scope,
                maximum_iterations=tf.reduce_max(self.word_length) * 2
            )
            # self.score = outputs.scores
            self.sample_id_preds = outputs.predicted_ids
            self.logits_preds = tf.no_op()
            self.result = tf.reshape(
                self.sample_id_preds[:, :, 0], [self._s[0], self._s[1], -1], name='result')

    def _add_loss_op(self):
        # assert self.logits
        # assert self.target_char_output
        # assert self.batch_size
        # assert self.target_word_length
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.target_char_output, logits=self.logits_train)
        target_weights = tf.sequence_mask(
            self.target_word_length, dtype=self.logits_train.dtype)
        self.loss += tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)
        tf.summary.scalar('loss', self.loss)

    def _build_model(self):
        self._add_placeholders()
        self._build_encoder()
        self._build_decoder()
        if self.mode == 'train':
            self._add_loss_op()
            self._add_train_op(method=self.configs.training_method,
                               loss=self.loss,
                               learning_rate=self.configs.learning_rate,
                               momentum=self.configs.momentum,
                               use_nesterov=self.configs.use_nesterov,
                               clip=self.configs.clip_grad)
            self._initialize_session()
            self._add_summary()

    def predict(self, texts):
        batch = self.configs.tokenizer.texts2matrix(texts, threshold=1.0)
        fd = {
            self.char_ids: batch['char_ids'],
            self.word_length: batch['word_length'],
            self.sequence_length: batch['sequence_length'],
            self.dropout: 1.0,
            self.start_token_id: self.configs.start_token_id,
            self.end_token_id: self.configs.end_token_id
        }
        res = self.sess.run(self.result, fd)
        return self.configs.tokenizer.matrix2texts(res)

    def train_loop(self,
                   batch,
                   dropout):

        fd = {
            self.char_ids: batch['char_ids'],
            self.word_length: batch['word_length'],
            self.sequence_length: batch['sequence_length'],
            self.target_char: batch['target_char'],
            self.target_char_len: batch['target_char_len'],
            self.dropout: dropout,
            self.start_token_id: self.configs.start_token_id,
            self.end_token_id: self.configs.end_token_id
        }
        run = self.sess.run(
            [self.train_op, self.train_summaries,
                self.loss, self.global_step], fd
        )
        self.logger.info("Step {}, loss {}".format(run[3], run[2]))

    def train_all_data(self, data_gen, dropout):
        for data in data_gen:
            self.train_loop(self.configs.tokenizer.texts2matrix(
                data, self.configs.threshold), dropout)
        self._save_model()


if __name__ == '__main__':
    import numpy as np
    a = np.array([[[1, 2, 0, 0], [3, 4, 5, 0], [0, 0, 0, 0]],
                  [[2, 1, 0, 0], [5, 4, 3, 0], [9, 7, 8, 6]]])
    l = np.array([[2, 3, 0], [2, 3, 4]])
    print(a)
    a[0][2:] = 0
    configs = Config()
    configs.nchars = 100
    configs.cdims = 100
    configs.encoder = {
        'num_hidden_char': [32, 32],
        'char_embedding': 'cnn',
        'char_embedding_kernel_size': [3, 3],
        'kernel_size': [3, 3],
        'num_filters': [64, 64],
        'dilation_rate': [1, 1],
        'gate': False,
        'concat': False
    }
    configs.decoder = {
        'num_layers': 2,
        'teacher': True
    }
    configs.set_log_dir('log_dir/spell')
    # with tf.Session() as sess:
    model = SpellingCorrector(configs=configs, mode='train')
    model._build_model()
    sess = model.sess
    print(sess.run(model.batch_size, feed_dict={model.char_ids: a}))
    m, n, k = sess.run([model.target_char_input, model.target_word_length, model.target_char_output], feed_dict={
        model.target_char: a,
        model.char_ids: a,
        model.target_char_len: l,
        model.start_token_id: 99,
        model.end_token_id: 32
    })
    print(m)
    print(n)
    print(k)
