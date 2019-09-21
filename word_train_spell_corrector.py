
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from model.spell_corrector_word import SpellingCorrector, Config, data_generator, Tokenizer


# In[2]:


data = pd.read_csv('../all_posts.csv')


# In[3]:


all_text = np.array(pd.concat([data['content'].dropna(), data['title'].dropna()]))


# In[4]:


tok = Tokenizer('all_char.txt','all_word.txt','<unk>')


# In[5]:


configs = Config()
configs.nchars = tok.num_chars() + 1
configs.cdims = 512
configs.char_embedding = 'rnn'
configs.num_hidden_char = [256]
configs.char_embedding_kernel_size = [3, 3, 3]
configs.nwords = tok.num_words() + 1
configs.use_cnn = True
configs.kernel_size = [5]
configs.num_filters = [256]
configs.dilation_rate = [1]
configs.gate_cnn = True
configs.num_hidden = [128]
configs.concat = True
configs.threshold = 0.9

configs.tokenizer = tok
configs.set_log_dir('log_dir/spell/word')
configs.version = 1
model = SpellingCorrector(configs=configs, mode='train')
tf.reset_default_graph()
model._build_model()
for _ in range(10):
    np.random.shuffle(all_text)
    i = 0
    while i < len(all_text):
        data_gen = data_generator(all_text[i:i+1024], 4, True)
        model.train_all_data(data_gen, 0.8)
        model.logger.info(model.predict(['cần mua nhà đừng lý thái tổ',
                             'cần mua nhà đơừng lý thi tổ',
                             'cần bán nhà quận bn thạnh',
                             'cần mua nhà đường nguyn đnh chỉu',
                             'cần mua nhà đường võ vă tn']))
        i += 1024

