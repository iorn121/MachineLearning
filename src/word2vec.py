import sys
sys.path.append('..')
import numpy as np

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
print(contexts,target)