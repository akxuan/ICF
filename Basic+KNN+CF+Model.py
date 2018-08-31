
# coding: utf-8

# In[2]:

from math import sqrt
import random
import time
def loadDataCF():
    trainSet = {}
    testSet = {}
    movieUser = {}
    u2u = {}
    dir_file ="C:/Users/Robert Chen/Downloads/ml-100k/" #this dir need to be changed
 
    TrainFile = dir_file+'u1.base'   #training file
    TestFile = dir_file+'u1.test'    #test file
    
    #load training file
    for line in open(TrainFile):
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        trainSet.setdefault(userId,{})
        trainSet[userId].setdefault(itemId,float(rating))
 
        movieUser.setdefault(itemId,[])
        movieUser[itemId].append(userId.strip())
    
    #load test file
    for line in open(TestFile): 
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        testSet.setdefault(userId,{})
        testSet[userId].setdefault(itemId,float(rating))
 
    #user to user matrix (both rated movies)
    for m in movieUser.keys():
        for u in movieUser[m]:
            u2u.setdefault(u,{})
            for n in movieUser[m]:
                if u!=n:
                    u2u[u].setdefault(n,[])
                    u2u[u][n].append(m)
    return trainSet,testSet,u2u

#calculate the average rating for each user
def getAverageRating(user):  
    average = (sum(trainSet[user].values())*1.0) / len(trainSet[user].keys())  
    return average

# Calibrate the similarity
def getUserSim(u2u,trainSet):
    userSim = {}
    # calculate user similarity
    for u in u2u.keys(): 
        userSim.setdefault(u,{})  
        average_u_rate = getAverageRating(u)  
        for n in u2u[u].keys():  
    
            userSim[u].setdefault(n,0)   #insert user n into the dic of user U
            average_n_rate = getAverageRating(n)  
            part1 = 0  #Pearson correlation coefficient Numerator 
            part2 = 0  #Pearson correlation coefficient denominator
            part3 = 0  #pearson correlation coefficient denominator
            for m in u2u[u][n]:  
                part1 += (trainSet[u][m]-average_u_rate)*(trainSet[n][m]-average_n_rate)*1.0  
                part2 += pow(trainSet[u][m]-average_u_rate, 2)*1.0  
                part3 += pow(trainSet[n][m]-average_n_rate, 2)*1.0  
        
            if part2 == 0 or part3 == 0:  #If the denominator is 0, the similarity is 0.
                userSim[u][n] = 0
            else:
                userSim[u][n] = part1 / (sqrt(part2) *sqrt(part3))
    return userSim

def getRecommendations(N,trainSet,userSim):
    pred = {}
    for user in trainSet.keys():    #for each user
        pred.setdefault(user,{})    #prediction dic format
        interacted_items = trainSet[user].keys() #User rated movies
        average_u_rate = getAverageRating(user)  #average rate for user rated movies
        userSimSum = 0
        simUser = sorted(userSim[user].items(),key = lambda x : x[1],reverse = True)[0:N]

        for n, sim in simUser:  
            average_n_rate = getAverageRating(n)
            userSimSum += sim   #calculate the similarity with neighbours
            for m, nrating in trainSet[n].items():  
                if m in interacted_items:  
                    continue  
                else:
                    pred[user].setdefault(m,0)
                    pred[user][m] += (sim * (nrating - average_n_rate))
        for m in pred[user].keys():  
                pred[user][m] = average_u_rate + (pred[user][m]*1.0) / userSimSum
    return pred

# Evaluate the model performance
def getMAE(testSet,pred):
    MAE = 0
    rSum = 0
    setSum = 0
 
    for user in pred.keys():    #For every user
        for movie, rating in pred[user].items():    #Every movie predicted for this user
            if user in testSet.keys() and movie in testSet[user].keys() : 
                #If the user rated the movie
                setSum = setSum + 1     #Predicting quantity+1
                rSum = rSum + abs(testSet[user][movie]-rating)      
                #Cumulative prediction error
    MAE = rSum / setSum
    return MAE


# In[4]:

# validated, ready to go
print("Basic CF model implementation")
if __name__ == '__main__':
    print("loading data")
    trainSet,testSet,u2u = loadDataCF()
    start = time.clock()
    print("calibrate user similarity")
    
    userSim = getUserSim(u2u,trainSet)
    end = time.clock()
    print("similarity calibration time： %f s" % (end - start))

    print("find nearest 5 user")
    for N in (5,10,20,30,40,50,60,70,80,90,100):        #for the number of searching neighbors K
        pred = getRecommendations(N,trainSet,userSim)   #get recommendation
        mae = getMAE(testSet,pred)  #calculate MAE
        print ('When N= %d, predictin accuracty：MAE=%f'%(N,mae))    






