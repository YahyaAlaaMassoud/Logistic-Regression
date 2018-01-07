import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)), (x)

def sigmoid_derivative(x, cache):
    activation,_ = sigmoid(cache)
    return x * activation * (1.0 - activation)

def relu(x):
    return np.maximum(0, x), (x)

def relu_derivative(x, cache):
    Z = cache
    dZ = np.array(x, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

#   image shape (length, height, depth)
#   output vec shape (length * height * depth, 1) 
#   len = shape[0], height = shape[1], depth = shape[2] 
#   axis = 1 -> return column vector
#   acis = 0 -> return row vector 
def img2vec(img, row_or_col = 2):
    if row_or_col == 0:
        vec = img.reshape(1, (img.shape[0] * img.shape[1] * img.shape[2]))
    elif row_or_col == 1:
        vec = img.reshape((img.shape[0] * img.shape[1] * img.shape[2]), 1)
    elif row_or_col == 2:
        vec = img.reshape((img.shape[0] * img.shape[1] * img.shape[2]))
    return vec

#def test_img2vec():
#    image = np.array([[[ 0.67826139,  0.29380381],
#            [ 0.90714982,  0.52835647],
#            [ 0.4215251 ,  0.45017551]],
#    
#           [[ 0.92814219,  0.96677647],
#            [ 0.85304703,  0.52351845],
#            [ 0.19981397,  0.27417313]],
#    
#           [[ 0.60659855,  0.00533165],
#            [ 0.10820313,  0.49978937],
#            [ 0.34144279,  0.94630077]]])
#    print ((img2vec(image, 2).shape))
#test_img2vec()

#   norm here computes the Euclidean/Forbenius Norm for every row(vector) in the matrix
#   and then divides the row by the value of the norm
#   axis = 0 -> columns
#   axis = 1 -> rows 
#   keepdims -> never change dimensions of the matrix after the operation 
def normalizeRows(x):
    x_forb_norm = np.linalg.norm(x, ord = None, axis = 1, keepdims = True)
    return x / x_forb_norm

def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis = 1, keepdims = True)

def L1_loss(y_hat, y):
    return np.sum(np.abs(y_hat - y))

def L2_loss(y_hat, y):
    return np.sum((y_hat - y)**2)