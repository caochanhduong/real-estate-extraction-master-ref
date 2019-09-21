
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from model.spell_corrector import SpellingCorrector, Config, Tokenizer, data_generator


# In[2]:


data = pd.read_csv('../all_posts.csv')


# In[3]:


all_text = np.array(pd.concat([data['content'].dropna(), data['title'].dropna()]))


# In[4]:


tok = Tokenizer('all_char.txt','<unk>')


# In[5]:


configs = Config()
configs.nchars = tok.num_chars() + 1
configs.tokenizer = tok
configs.cdims = 256
configs.set_log_dir('log_dir/spell')
configs.version = 1
configs.start_token_id = tok.dictionary['<sos>']
configs.end_token_id = 0
configs.encoder = {
    'num_hidden_char': [128],
    'char_embedding': 'rnn',
    'char_embedding_kernel_size': [3, 3],
    'kernel_size': [3, 3],
    'num_filters': [128, 128],
    'dilation_rate': [1, 1],
    'concat': False
}
configs.decoder = {
    'num_layers': 1,
    'teacher': True,
    
}
model = SpellingCorrector(configs=configs, mode='train')
tf.reset_default_graph()
model._build_model()
for _ in range(10):
    np.random.shuffle(all_text)
    i = 0
    while i < len(all_text):
        data_gen = data_generator(all_text[i:i+1024], 16, True)
        model.train_all_data(data_gen, 0.8)
        print(model.predict(['cần mua nhà đừng lý thái tổ','cần bán nhà quận b thạnh','cần mua nhà đường nguyn đnh chỉu']))
        i += 1024

