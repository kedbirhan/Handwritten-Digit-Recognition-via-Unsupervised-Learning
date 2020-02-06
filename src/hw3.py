from copy import deepcopy
import numpy as np
import pandas as pd
import random
import math
from scipy.spatial import distance
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plotter(dic,centroid):
    colors = ['magenta', 'red', 'lime', 'cyan', 'orange', 'gray','black', 'blue', 'purple', 'yellow']

    plt.figure(figsize=(20,15))
    c=0
    for i in range(len(dic)):
        length=len(dic[c])
        c=c+1
        li=np.array(dic[i])
        x=len(li)
        y=x
        plt.scatter(li[:,0], li[:,1], s=10, c=colors[i])
    j=0
    z=1
    e=9
    for i in range(len(colors)):
        plt.scatter(centroid[i][j], centroid[i][z], s=300, c=colors[e])
        e=e-1

def K_means(data,centroids,k,tot=0.01,max=300):
    centroid=deepcopy(centroids) # copy the centroid to  a new variables
    classfication={} #thsi will holds the centroids and it's point 0:[], 1:[], 2:[]......
    index_array=[]  # holds the index 
    counter=0  # keeps tracks of how many times the centroid is recalculated

#     for i in range(max):
    optimized=False
    while optimized!=True:
        counter=counter+1
        # initalize the dict: works for any number of k that we want to intialze nice move including k in the paramter
        for j in range(k):
            classfication[j]=[]  # initalize empty dictionary for the each cluster
        index_array=[] 
       
    #for each datapoint find in which cluster it belongs and insert it in its   it to that cluster
        for x in data:
           # distances = [round(np.linalg.norm(x-centroid[y]),2)for y in range(k)]  # distance of x between each centroid
            distances=[round(distance.euclidean(x,centroid[y]),1)for y in range(k)]
            type=distances.index(min(distances))
            index_array.append(type+1)
            classfication[type].append(x)    # append the position
#         print(len(classfication))
        #get the new centroid
#         print(classfication)
        new_centroid=[]  # hold the new centroid

        for x in range(k):       
#             new_centroid[x]=np.avergae(classfication[x],axis=1)
# why zero is becase this is a high dimentional data and we want the average of each 
# columns to determine the centroid
             new_centroid.append(np.average(classfication[x],axis=0))
        optimized=True
        centroid_dis=[]  # array to hold the distance of the current and the new centroid
        # find the distance between old centroid and new centroid
        # for c in range(k):
        #     old_cen=centroid[c]
        #     new_cen=new_centroid[c]
        #     centroid_dis.append(np.linalg.norm(new_cen-old_cen,axis=0))

          # see if the old and the new centroid is within 0.01   
        # for i in range(k):
        #     if centroid_dis[i]>=tot:
        #         optimized=False
        #         centroid=deepcopy(new_centroid)
        for i in range(k):
            centroid_dis.append(distance.euclidean(centroid[i], new_centroid[i]))
        for i in range(k):
            if centroid_dis[i]>0.01:
                optimized=False
                break
        centroid=deepcopy(new_centroid)
    write_to_file(index_array)
    plotter(classfication,centroids)
    plotter(classfication,centroid)
    print("done with kmeans \n number of times  optimzed is "+ str(counter))
#     return index_array classfication
    return centroid

def pickCentroid(items,k,size):
    index=random.sample(range(size),k)
    print(index)
    item=[]
    
    for i in range(k):
        j=index[i]
        h=items[j]
        item.append(h)
    return item

def write_to_file(array):
    # j=1
    # increment=[i+j for i in array]
    write=open("submit.txt","w")
    write.writelines("%s\n" % int(i) for i in array)
    write.close()

data = pd.read_csv('test.txt', sep=",",header=None)
print(data.shape) #must be inside print to see the shape sice it is a tubple
data.head()
k=10
print("done feating data")

# y=PCA(n_components=3).fit_transform(data)
y=manifold.Isomap(n_neighbors=5,n_components=2).fit_transform(data)
print("done manifold")
centroid=pickCentroid(y,k,len(y))
cen= K_means(y,centroid,k)
print("all done")