import sys
sys.path.append('..')
import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def preprocess(text):
  text = text.lower()
  text = text.replace('.',' .')
  words = text.split(' ')

  word_to_id = {}
  id_to_word = {}
  for word in words:
    if word not in word_to_id:
      new_id=len(word_to_id)
      word_to_id[word] = new_id
      id_to_word[new_id] = word

  corpus = np.array([word_to_id[word] for word in words])

  return corpus,word_to_id,id_to_word

class MatMul:
  def __init__(self,W):
    self.params = [W]
    self.grads = [np.zeros_like(W)]
    self.x = None

  def forward(self,x):
    W = self.params
    out = np.dot(x,W)
    self.x = x
    return out

  def backward(self,dout):
    W, = self.params
    dx =np.dot(dout,W.T)
    dW = np.dot(self.x.T,dout)
    self.grads[0][...] = dW
    return dx

def create_contexts_target(corpus,window_size=1):
  target= corpus[window_size:-window_size]
  contexts=[]

  for idx in range(window_size,len(corpus)-window_size):
    cs=[]
    for t in range(-window_size,window_size+1):
      if(t==0):
        continue
      cs.append(corpus[idx+t])
    contexts.append(cs)
  return np.array(contexts), np.array(target)

def convert_one_hot(corpus, vocab_size):

  N = corpus.shape[0]

  if corpus.ndim == 1:
    one_hot = np.zeros((N, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
      one_hot[idx, word_id] = 1

  elif corpus.ndim == 2:
    C = corpus.shape[1]
    one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
    for idx_0, word_ids in enumerate(corpus):
      for idx_1, word_id in enumerate(word_ids):
        one_hot[idx_0, idx_1, word_id] = 1

  return one_hot

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx



c0=np.array([[1,0,0,0,0,0,0]])
c1=np.array([[0,0,1,0,0,0,0]])

W_in = np.random.randn(7,3)
W_out = np.random.randn(3,7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5*(h0+h1)
s=out_layer.forward(h)

text = 'You say goodbye and I say hello.'
corpus,word_to_id,id_to_word=preprocess(text)

contexts, target = create_contexts_target(corpus,window_size=1)
vocab_size = len(word_to_id)
target=convert_one_hot(target,vocab_size)
contexts=convert_one_hot(contexts,vocab_size)

class SimpleCBOW:
  def __init__(self,vocab_size,hidden_size):
    V,H=vocab_size,hidden_size

    # 重みの初期化
    W_in=0.01*np.random.randn(V,H).astype('f')
    W_out=0.01*np.random.randn(V,H).astype('f')

    # レイヤの作成
    self.in_layer0=MatMul(W_in)
    self.in_layer1=MatMul(W_in)
    self.out_layer=MatMul(W_out)
    self.loss_layer=SoftmaxWithLoss()

    # 全ての重みと勾配をリストにまとめる
    layers=[self.in_layer0,self.in_layer1,self.out_layer]
    self.params,self.grads=[],[]
    for layer in layers:
      self.params+=layer.params
      self.grads+=layer.grads

    self.word_vecs=W_in
    
  def forward(self,contexts,target):
    h0=self.in_layer0.forward(contexts[:,0])
    h1=self.in_layer0.forward(contexts[:,1])
    h=(h0+h1)*0.5
    score=self.out_layer.forward(h)
    loss=self.loss_layer.forward(score,target)
    return loss
  
  def backward(self,dout=1):
    ds= self.loss_layer.backward(dout)
    da=self.out_layer.backward(ds)
    da*=0.5
    self.in_layer0.backward(da)
    self.in_layer1.backward(da)
    return None
  