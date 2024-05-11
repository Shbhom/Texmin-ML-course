#copied from colab so will have the pip install command
!pip install numpy

##LR implementation
import numpy as np

class LR:
  def __init__(self,N):
    self.theta = np.zeros(N)

  def train(self,x,y):
    np_x = np.array(x)
    np_y = np.array(y)
    np_x_trans = np_x.T
    x_trans_dot_x= np.dot(np_x_trans,np_x)
    print("x_trans_dot_x",x_trans_dot_x)
    inversed_value = np.linalg.inv(x_trans_dot_x)
    print("inversed_value",inversed_value)
    x_trans_dot_y = np.dot(np_x_trans,np_y)
    final_dot = np.dot(inversed_value,x_trans_dot_y)
    self.theta = final_dot

  def predict(self,x):
    np_x = np.array(x)
    return np.dot(np_x,self.theta)

##testing the code
test = LR(2)
test.train([[1,1],[2,3]],[4,8])
print(test.predict([7,8]))
