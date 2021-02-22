import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#K Nearest Neighbours Algorithm

#Function to calculate eucledian distance
def eucledian_distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

#KNN function
def knn(x_train,y_train,test_point,k=5):
    #1. make an empty distance list and fill that list with distances from all points from training set.
    distances=[]
    for point,label in zip(x_train,y_train):
        dist=eucledian_distance(point,test_point)
        distances.append((dist,label))
    #2. sort on basis of distances
    distances=sorted(distances, key=lambda x:x[0])
    distances=np.array(distances)
    #3. select k nearest neighbors
    distances=distances[:k]
    
    #4. get the lebel counts
    freq=np.unique(distances[:,1],return_counts=True)
    labels,counts=freq
    ans=labels[counts.argmax()]
    
    return ans

#Function to calculate predictions of all points in dataset
def get_all_predictions(x_train,y_train,x_test,k=5):
    predictions=[]
    for test_point in x_test:
        test_label=knn(x_train,y_train,test_point,k)
        predictions.append(test_label)
    return predictions

#Function to calculate accuracy of our algorithm
def get_accuracy(predictions,true_labels):
    return (predictions==true_labels).sum()/true_labels.shape[0]
    
 #KNN on MNIST dataset

#Make dataframe of mnist dataset
df=pd.read_csv("mnist/train.csv")
df.head()

#MNIST dataset size id (42000,785) i.e it has 784 pixel values of 42000 images of digits
df=df.values
print(df)
print(df.shape)

#5000 values of the dataset are selected 
df=df[:5000]
print(df.shape)

#80% data is used for training and remaining is used for testing
split=(int)(0.8*df.shape[0])

#1st column of dataframe contains lable values and remaing columns contain 784 pixel values
mnist_x_train=df[:split,1:]
mnist_y_train=df[:split,0]

mnist_x_test=df[split:,1:]
mnist_y_test=df[split:,0]

print(mnist_x_train.shape)
print(mnist_y_train.shape)
print(mnist_x_test.shape)
print(mnist_y_test.shape)

#Predictions are made on testing data using KNN agorithm
mnist_all_predictions=get_all_predictions(mnist_x_train,mnist_y_train,mnist_x_test,k=5)

#Reshape the dataset into (28,28) image using pixel values and print the image
test_examples=mnist_x_test[200:240,:]
test_labels=mnist_y_test[200:240]

for test_point,test_label in zip(test_examples,test_labels):
    my_pred=knn(mnist_x_train,mnist_y_train,test_point,k=5)
    img=test_point.reshape(28,28)
    plt.imshow(img,cmap="gray")
    plt.show()
    print("predicted label",int(my_pred))
    print("true label", test_label)
    print("***************************************************************************************************")

print()
print()

#Accuracy calculation
print("Accuracy of our algorithm is ",get_accuracy(mnist_all_predictions,mnist_y_test))

#KNN on Fashion MNIST dataset

#Make dataframe of mnist dataset
df=pd.read_csv("Fashion Dataset/train.csv")
#df.head()

#MNIST dataset size id (60000,785) i.e it has 784 pixel values of 60000 images of articles
df=df.values
print(df)
print(df.shape)

#5000 values of the dataset are selected 
df=df[:5000]
print(df.shape)

#80% data is used for training and remaining is used for testing
split=(int)(0.8*df.shape[0])

#1st column of dataframe contains lable values and remaing columns contain 784 pixel values
mnist_x_train=df[:split,1:]
mnist_y_train=df[:split,0]

mnist_x_test=df[split:,1:]
mnist_y_test=df[split:,0]

print(mnist_x_train.shape)
print(mnist_y_train.shape)
print(mnist_x_test.shape)
print(mnist_y_test.shape)

#Names of labels
dict_labels={0:"T-shirt/top",
1:"Trouser",
2:"Pullover",
3:"Dress",
4:"Coat",
5:"Sandal",
6:"Shirt",
7:"Sneaker",
8:"Bag",
9:"Ankle boot"}
print(dict_labels)

#Predictions are made on testing data using KNN agorithm
mnist_all_predictions=get_all_predictions(mnist_x_train,mnist_y_train,mnist_x_test,k=5)

#Reshape the dataset into (28,28) image using pixel values and print the image
test_examples=mnist_x_test[200:240,:]
test_labels=mnist_y_test[200:240]

for test_point,test_label in zip(test_examples,test_labels):
    my_pred=knn(mnist_x_train,mnist_y_train,test_point,k=5)
    img=test_point.reshape(28,28)
    plt.imshow(img,cmap="gray")
    plt.show()
    pred_name=dict_labels[int(my_pred)]
    actual_name=dict_labels[test_label]
    print("predicted label:", pred_name)
    print("true label:", actual_name)
    print("***************************************************************************************************")

print()
print()

#Accuracy calculation
print("Accuracy of our algorithm is ",get_accuracy(mnist_all_predictions,mnist_y_test))
