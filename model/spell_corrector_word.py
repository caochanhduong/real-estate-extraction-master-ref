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
    def __init__(self, char_file, word_file, oov_token='<unk>', pad_val=0):
        with open(char_file, 'r') as _in:
            self.char_dictionary = {x: int(i) for x, i in map(
                lambda x: x.strip().split('\t'), _in.readlines())}
            if self.char_dictionary.get(oov_token) is None:
                self.char_dictionary[oov_token] = len(self.char_dictionary)
        with open(word_file, 'r') as _in:
            self.word_dictionary = {x: int(i) for x, i in map(
                lambda x: x.strip().split('\t'), _in.readlines())}
            if self.word_dictionary.get(oov_token) is None:
                self.word_dictionary[oov_token] = len(self.word_dictionary)
        self.oov_token = oov_token
        self.pad_val = pad_val
        self.idx2char = {i: x for x, i in self.char_dictionary.items()}
        self.idx2word = {i: x for x, i in self.word_dictionary.items()}

    def charmatrix2texts(self, matrix):
        return [
            ' '.join(''.join(self.idx2char[x] for x in word if x > 0) for word in line).strip() for line in matrix
        ]

    def wordmatrix2texts(self, matrix):
        return [
            ' '.join(self.idx2word[x] for x in line if x > 0) for line in matrix
        ]

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

    def texts2matrixword(self, texts):
        words = [self._clean_text(x).split() for x in texts]
        seq_len = np.array([len(x) for x in words])
        padded_seq = np.zeros([len(texts), seq_len.max()], dtype=np.int32)
        for i in range(len(texts)):
            for j in range(len(words[i])):
                padded_seq[i][j] = self.word_dictionary.get(
                    words[i][j], self.word_dictionary[self.oov_token])
        return {
            'tokens': padded_seq,
            'sequence_length': seq_len
        }

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
        words = [self.noise_maker(x, threshold) for x in words]
        res = []
        word_len = []
        for word in words:
            word_len.append(len(word))
            res.append([self.char_dictionary.get(
                x, self.char_dictionary[self.oov_token]) for x in list(word)])
        return res, word_len

    def texts2matrixchar(self, texts, threshold=1.0):
        arrays = []
        word_len = []
        seq_len = []
        for text in texts:
            arr, wl = self.text2array(text, threshold)
            arrays.append(arr)
            word_len.append(wl)
            seq_len.append(len(wl))
        seq_len = np.array(seq_len, dtype=np.int32)
        padded_word_len = np.zeros(
            [len(word_len), seq_len.max()], dtype=np.int32)
        for i in range(len(word_len)):
            padded_word_len[i][:len(word_len[i])] = word_len[i]
        max_word_len = np.max(padded_word_len)
        padded_arrays = np.zeros(
            [len(arrays), seq_len.max(), max_word_len], dtype=np.int32)
        for i in range(len(arrays)):
            for j in range(len(arrays[i])):
                padded_arrays[i][j][:len(arrays[i][j])] = arrays[i][j]
        return {
            'char_ids': padded_arrays,
            'word_length': padded_word_len,
            'sequence_length': seq_len
        }

    def text2data(self, texts, threshold):
        a = self.texts2matrixchar(texts, threshold)
        a.update(self.texts2matrixword(texts).items())
        return a

    def num_chars(self):
        return len(self.char_dictionary)

    def num_words(self):
        return len(self.word_dictionary)


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
        self.tokenizer = None
        self.version = 1
        self.wdims = None
        self.nwords = None
        self.gate_cnn = False
        self.num_hidden_char = None
        self.char_embedding_kernel_size = None
        self.use_cnn = True
        self.kernel_size = None
        self.num_filters = None
        self.dilation_rate = None
        self.num_hidden = None
        self.concat = True
        self.threshold = 0.9

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
        # self.start_token_id = tf.placeholder(
        #     tf.int32, [], name='start_token_id'
        # )
        # self.end_token_id = tf.placeholder(
        #     tf.int32, [], name='end_token_id'
        # )
        self._s = tf.shape(self.char_ids)
        self.batch_size = self._s[0]*self._s[1]
        self.loss = 0
        if self.mode == 'train':
            self.target_word = tf.placeholder(
                tf.int32, [None, None], name='target_word')
            self.dropout = tf.placeholder(tf.float32, [], name='dropout')
            # self.learning_rate = tf.placeholder(
            # tf.float32, [], name='learning_rate')
            self.rnn_cell = lambda hidden: build_gru_cell_with_dropout(
                hidden, self.dropout)
        else:
            self.dropout = None
            self.rnn_cell = build_gru_cell

    def _build_word_embedding(self):
        with tf.variable_scope('char_embedding'):
            char_embedding = tf.Variable(
                tf.random_uniform(
                    [self.configs.nchars, self.configs.cdims],
                    -0.1, 0.1,
                    tf.float32
                ),
                name='embedding_matrix'
            )
            char_embedding = tf.nn.embedding_lookup(
                char_embedding, self.char_ids,
                name='lookup'
            )
            s = tf.shape(char_embedding)
            char_embedding = tf.reshape(
                char_embedding,
                [s[0]*s[1], s[2], self.configs.cdims]
            )
            if self.configs.char_embedding == 'rnn':
                with tf.variable_scope('char_rnn'):
                    word_length = tf.reshape(self.word_length, [s[0]*s[1]])
                    _, fs_fw, fs_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        [self.rnn_cell(x)
                            for x in self.configs.num_hidden_char],
                        [self.rnn_cell(x)
                            for x in self.configs.num_hidden_char],
                        char_embedding,
                        sequence_length=word_length,
                        dtype=tf.float32
                    )
                    output = tf.concat([fs_fw[-1], fs_bw[-1]], axis=-1)
                    output = tf.nn.relu(output)
                    final_size = self.configs.num_hidden_char[-1] * 2
            else:
                with tf.variable_scope('char_cnn'):
                    word_length = tf.reshape(
                        self.word_length, [self.batch_size])
                    mask = tf.expand_dims(tf.sequence_mask(word_length, dtype=tf.float32),
                                          axis=-1)
                    output = char_embedding
                    for kernel_size, num_filters in zip(self.configs.char_embedding_kernel_size, self.configs.num_hidden_char):
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
                    final_size = self.configs.num_hidden_char[-1]
            output = tf.reshape(output, [s[0], s[1], final_size])
            self.encoder_word_emb = tf.nn.dropout(
                x=output, keep_prob=self.dropout, noise_shape=[s[0], 1, output.get_shape()[2]]) if self.mode == 'train' else output

    def _build_main_part(self):
        if self.configs.use_cnn:
            with tf.variable_scope('main_cnn'):
                mask = tf.expand_dims(tf.sequence_mask(self.sequence_length, dtype=tf.float32),
                                      axis=-1)
                output = self.encoder_word_emb
                for ksz, filters, dl in zip(self.configs.kernel_size, self.configs.num_filters, self.configs.dilation_rate):
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
                    if self.configs.gate_cnn:
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
            with tf.variable_scope('main_rnn'):
                outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    [self.rnn_cell(x)
                     for x in self.configs.num_hidden],
                    [self.rnn_cell(x)
                     for x in self.configs.num_hidden],
                    self.encoder_word_emb,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32
                )
        with tf.variable_scope('output'):
            if self.configs.concat:
                output = tf.concat([output, self.encoder_word_emb], -1)
            output = output * tf.stop_gradient(tf.tile(mask,
                                                       multiples=[1, 1, output.get_shape()[-1]]))
            self.logits = tf.layers.conv1d(
                inputs=output,
                filters=self.configs.nwords,
                kernel_size=1,
                padding='valid',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.nn.relu,
                name='logits'
            )
        self.result = tf.argmax(
            self.logits, axis=-1, output_type=tf.int32, name='result')

    def _add_loss_op(self):
        # assert self.logits
        # assert self.target_char_output
        # assert self.batch_size
        # assert self.target_word_length
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.target_word, logits=self.logits)
        target_weights = tf.sequence_mask(self.sequence_length)
        self.loss += tf.reduce_mean(tf.boolean_mask(tensor=crossent,
                                                    mask=target_weights))
        tf.summary.scalar('loss', self.loss)

    def _build_model(self, pretrained_word_embedding=None, pretrained_char_embedding=None):
        self._add_placeholders()
        self._build_word_embedding()
        self._build_main_part()
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
        batch = self.configs.tokenizer.texts2matrixchar(texts, threshold=1.0)
        fd = {
            self.char_ids: batch['char_ids'],
            self.word_length: batch['word_length'],
            self.sequence_length: batch['sequence_length'],
            self.dropout: 1.0
        }
        res = self.sess.run(self.result, fd)
        return self.configs.tokenizer.wordmatrix2texts(res)

    def train_loop(self,
                   batch,
                   dropout):
        fd = {
            self.char_ids: batch['char_ids'],
            self.word_length: batch['word_length'],
            self.sequence_length: batch['sequence_length'],
            self.target_word: batch['tokens'],
            self.dropout: dropout
        }
        run = self.sess.run(
            [self.train_op, self.train_summaries,
                self.loss, self.global_step], fd
        )
        self.logger.info("Step {}, loss {}".format(run[3], run[2]))

    def train_all_data(self, data_gen, dropout):
        for data in data_gen:
            self.train_loop(
                self.configs.tokenizer.text2data(data, threshold=self.configs.threshold), dropout)
        self._save_model()


if __name__ == '__main__':
    import numpy as np
    a = np.array([[[1, 2, 0, 0], [3, 4, 5, 0], [0, 0, 0, 0]],
                  [[2, 1, 0, 0], [5, 4, 3, 0], [9, 7, 8, 6]]])
    l = np.array([[2, 3, 0], [2, 3, 4]])
    t = np.array([[2, 3, 0], [4, 5, 7]])
    print(a)
    a[0][2:] = 0
    configs = Config()
    configs.nchars = 100
    configs.cdims = 100
    configs.nwords = 100

    with tf.Session() as sess:
        model = SpellingCorrector(configs=configs, mode='train')
        model._build_model()
        print(sess.run(model.batch_size, feed_dict={model.char_ids: a}))
        m, n, k = sess.run([model.target_word_input, model.target_word_output, model.target_sequence_length], feed_dict={
            model.char_ids: a,
            model.word_length: l,
            model.start_token_id: 99,
            model.end_token_id: 32,
            model.target_word: t})
        print(m)
        print(n)
        print(k)
