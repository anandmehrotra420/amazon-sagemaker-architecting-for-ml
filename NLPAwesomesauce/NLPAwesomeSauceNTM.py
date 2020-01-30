#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import sagemaker
from sagemaker import get_execution_role
import s3fs
import time
import boto3, os
from urllib.parse import urlparse

sess = sagemaker.Session()
run_prefix = str(int(round(time.time() * 1000)))
role = get_execution_role()


# In[5]:


#read csv from s3 bucket

df = pd.read_csv('s3://nlp-awesome-sauce/dataset/lowtrain.csv', usecols = [3])
#low_df_test = pd.read_csv('s3://nlp-awesome-sauce/dataset/lowtrain.csv',names=["rating", "title", "review"])
#low_df_test[low_df_test['review'].str.contains('vacuum')]
df.head() 


# In[31]:


df.to_csv("reviewsOnly2.csv", header=False)
get_ipython().system('aws s3 cp reviewsOnly2.csv s3://nlp-awesome-sauce/dataset/ntm/Jan30/')


# In[35]:


df = df.sample(frac=.5)


# In[36]:


data = df.to_numpy().flatten()


# In[37]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import re
token_pattern = re.compile(r"(?u)\b\w\w+\b")
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if len(t) >= 2 and re.match("[a-z].*",t) 
                and re.match(token_pattern, t)]


# In[38]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vocab_size = 2000
print('Tokenizing and counting, this may take a few minutes...')
start_time = time.time()
print("Start. {}".format(start_time))
vectorizer = CountVectorizer(input='content', analyzer='word', stop_words='english',
                             tokenizer=LemmaTokenizer(), max_features=vocab_size, max_df=0.95, min_df=2)
vectors = vectorizer.fit_transform(data)
vocab_list = vectorizer.get_feature_names()
print('vocab size:', len(vocab_list))

# random shuffle
idx = np.arange(vectors.shape[0])
np.random.shuffle(idx)
vectors = vectors[idx]

print('Done. Time elapsed: {:.2f}s'.format(time.time() - start_time))


# In[39]:


import scipy.sparse as sparse
vectors = sparse.csr_matrix(vectors, dtype=np.float32)
print(type(vectors), vectors.dtype)


# In[40]:


n_train = int(0.8 * vectors.shape[0])

# split train and test
train_vectors = vectors[:n_train, :]
test_vectors = vectors[n_train:, :]

# further split test set into validation set (val_vectors) and test  set (test_vectors)
n_test = test_vectors.shape[0]
val_vectors = test_vectors[:n_test//2, :]
test_vectors = test_vectors[n_test//2:, :]

print(train_vectors.shape, test_vectors.shape, val_vectors.shape)


# In[52]:


import os
from sagemaker import get_execution_role

role = get_execution_role()

bucket = 'nlp-awesome-sauce'
prefix = 'ntm'

train_prefix = os.path.join(prefix, 'train', 'Jan30')
val_prefix = os.path.join(prefix, 'val', 'Jan30')
test_prefix = os.path.join(prefix, 'test', 'Jan30')
output_prefix = os.path.join(prefix, 'output', 'Jan30')

s3_train_data = os.path.join('s3://', bucket, train_prefix)
s3_val_data = os.path.join('s3://', bucket, val_prefix)
s3_test_data = os.path.join('s3://', bucket, test_prefix)
output_path = os.path.join('s3://', bucket, output_prefix)
print('Training set location', s3_train_data)
print('Validation set location', s3_val_data)
print('Trained model will be saved at', output_path)


# In[42]:


def split_convert_upload(sparray, bucket, prefix, fname_template='data_part{}.pbr', n_parts=2):
    import io
    import sagemaker.amazon.common as smac
    
    chunk_size = sparray.shape[0]// n_parts
    for i in range(n_parts):

        # Calculate start and end indices
        start = i*chunk_size
        end = (i+1)*chunk_size
        if i+1 == n_parts:
            end = sparray.shape[0]
        
        # Convert to record protobuf
        buf = io.BytesIO()
        smac.write_spmatrix_to_sparse_tensor(array=sparray[start:end], file=buf, labels=None)
        buf.seek(0)
        
        # Upload to s3 location specified by bucket and prefix
        fname = os.path.join(prefix, fname_template.format(i))
        boto3.resource('s3').Bucket(bucket).Object(fname).upload_fileobj(buf)
        print('Uploaded data to s3://{}'.format(os.path.join(bucket, fname)))


# In[53]:


split_convert_upload(train_vectors, bucket=bucket, prefix=train_prefix, fname_template='train_part{}.pbr', n_parts=8)


# In[54]:


split_convert_upload(val_vectors, bucket=bucket, prefix=val_prefix, fname_template='val_part{}.pbr', n_parts=1)


# In[55]:


split_convert_upload(test_vectors, bucket=bucket, prefix=test_prefix, fname_template='test_part{}.pbr', n_parts=1)


# In[82]:


containers = {'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/ntm:latest'}
ntm = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                    role, 
                                    train_instance_count=4, 
                                    train_instance_type='ml.c4.xlarge',
                                    output_path=output_path,
                                    sagemaker_session=sess)
num_topics = 6
ntm.set_hyperparameters(num_topics=num_topics, feature_dim=vocab_size, mini_batch_size=128, 
                        epochs=100, num_patience_epochs=5, tolerance=0.001)


# In[83]:


from sagemaker.session import s3_input
s3_train = s3_input(s3_train_data, distribution='ShardedByS3Key') 


# In[84]:


ntm.fit({'train': s3_train, 'test': s3_val_data})


# In[59]:


ntm_predictor = ntm.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# In[65]:


from sagemaker.predictor import csv_serializer, json_deserializer

ntm_predictor.content_type = 'text/csv'
ntm_predictor.serializer = csv_serializer
ntm_predictor.deserializer = json_deserializer

test_data = np.array(test_vectors.todense())
results = ntm_predictor.predict(test_data[:5])
predictions = np.array([prediction['topic_weights'] for prediction in results['predictions']])
print(predictions)


# In[60]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fs = 12
df=pd.DataFrame(predictions.T)
df.plot(kind='bar', figsize=(16,4), fontsize=fs)
plt.ylabel('Topic assignment', fontsize=fs+2)
plt.xlabel('Topic ID', fontsize=fs+2)


# In[61]:


import pandas as pd
import os
get_ipython().system('pip install mxnet ')
import mxnet as mx
get_ipython().system('pip install WordCloud')
from wordcloud import WordCloud


# In[85]:


model_path = os.path.join('ntm/output/Jan30/ntm-2020-01-30-19-16-39-306/output/model.tar.gz')
print(model_path)


# In[86]:


import boto3
boto3.resource('s3').Bucket('nlp-awesome-sauce').download_file(model_path, 'downloaded_model.tar.gz')


# In[87]:


get_ipython().system("tar -xzvf 'downloaded_model.tar.gz'")
# use flag -o to overwrite previous unzipped content
get_ipython().system('unzip -o model_algo-1')
model = mx.ndarray.load('params')
W = model['arg:projection_weight']


# In[ ]:


##visualize?
word_to_id = dict()
for i, v in enumerate(vocab_list):
    word_to_id[v] = i

limit = 24
n_col = 4
counter = 0

plt.figure(figsize=(20,16))
for ind in range(num_topics):

    if counter >= limit:
        break

    title_str = 'Topic{}'.format(ind)

    #pvals = mx.nd.softmax(W[:, ind]).asnumpy()
    pvals = mx.nd.softmax(mx.nd.array(W[:, ind])).asnumpy()

    word_freq = dict()
    for k in word_to_id.keys():
        i = word_to_id[k]
        word_freq[k] =pvals[i]

    wc = WordCloud(background_color='white').fit_words(word_freq)

    plt.subplot(limit // n_col, n_col, counter+1)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title_str)
    #plt.close()

    counter +=1


# In[81]:




pd.DataFrame(vocab_list).to_csv('vocab.csv')

get_ipython().system('aws s3 cp vocab.csv s3://nlp-awesome-sauce/dataset/ntm/')


# In[82]:




sagemaker.Session().delete_endpoint(ntm_predictor.endpoint)


# In[ ]:




