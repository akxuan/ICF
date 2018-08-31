
# coding: utf-8

# In[162]:

from math import sqrt
import random
import time

dir_file ="C:/Users/Robert Chen/Downloads/ml-100k/"  #this dir need to be changed

 
def loadData():
    trainSet = {} # ramdomly 60K rows
    trainSet5K_1 = {}
    trainSet5K_2 = {}
    trainSet5K_3 = {}
    testSet = {}
    
    movieUser = {}
    u2u = {}

     
    TrainFile = dir_file+'u1.base'   #training file
    TestFile = dir_file+'u1.test'    #testing file
    
    #加载训练集
    Train_file_random = []
    for line in open(TrainFile):
        Train_file_random.append(line)
        
    random.shuffle(Train_file_random)
    Train_file_random60K = Train_file_random[:65000]
    Train_file_random5K_1 = Train_file_random[65001:70000]
    Train_file_random5K_2 = Train_file_random[70001:75000]
    Train_file_random5K_3 = Train_file_random[75001:]
    
    for line in Train_file_random60K:
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        trainSet.setdefault(userId,{})
        trainSet[userId].setdefault(itemId,float(rating))
        movieUser.setdefault(itemId,[])
        movieUser[itemId].append(userId.strip())    
    
    for line in Train_file_random5K_1:
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        trainSet5K_1.setdefault(userId,{})
        trainSet5K_1[userId].setdefault(itemId,float(rating))
    
    for line in Train_file_random5K_2:
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        trainSet5K_2.setdefault(userId,{})
        trainSet5K_2[userId].setdefault(itemId,float(rating))
    
    for line in Train_file_random5K_3:
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        trainSet5K_3.setdefault(userId,{})
        trainSet5K_3[userId].setdefault(itemId,float(rating))
        
    #load test file
    for line in open(TestFile): 
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        testSet.setdefault(userId,{})
        testSet[userId].setdefault(itemId,float(rating))
 
    #Generate User to User Movie List
    for m in movieUser.keys():
        for u in movieUser[m]:
            u2u.setdefault(u,{})
            for n in movieUser[m]:
                if u!=n:
                    u2u[u].setdefault(n,[])
                    u2u[u][n].append(m)
    
    return trainSet,trainSet5K_1,trainSet5K_2,trainSet5K_3,testSet,movieUser,u2u


# In[163]:

#calculate the average rating for each user
def getAverageRating(user):  
    average = (sum(trainSet[user].values())*1.0) / len(trainSet[user].keys())  
    return average


# In[164]:

def getUserSim_full(u2u,trainSet):
    
    userSim_full = {}
    # calculate user similarity
    for u in u2u.keys(): 
        userSim_full.setdefault(u,{}) 
        part4 = len(trainSet[u])  # num of item user u has rated
        part5 = average_u_rate = getAverageRating(u)
         
          #average rating of user u
        
        for n in u2u[u].keys():               
                        
            userSim_full[u].setdefault(n,())  #insert user n into the dic of user U
            part1 = 0  #Pearson correlation coefficient Numerator 
            part2 = 0  #Pearson correlation coefficient denominator
            part3 = 0  #pearson correlation coefficient denominator
            part6 = average_n_rate = getAverageRating(n)
            part7 = 0
            part8 = 0
        
            for m in u2u[u][n]:  #for user u & user n, all movies with both rated  
                part1 += (trainSet[u][m]-average_u_rate)*(trainSet[n][m]-average_n_rate)*1.0  
                part2 += pow(trainSet[u][m]-average_u_rate, 2)*1.0  
                part3 += pow(trainSet[n][m]-average_n_rate, 2)*1.0 
                part7 += trainSet[n][m]
                part8 += trainSet[u][m]
            
            if part2 == 0 or part3 == 0:  #If the denominator is 0, the similarity is 0.
                A = 0
            else:
                A = part1 / (sqrt(part2) * sqrt(part3))
            userSim_full[u][n] = (A,part1,part2,part3,part4,part5,part6,part7,part8)   
        
    return userSim_full


# In[165]:

# update user similarity table by one user input
def updateUserSim_full(rate_in,userSim_full,trainSet):
    user_in  = rate_in[0]
    movie_in = rate_in[1]
    rate_in  = rate_in[2]

    if user_in not in trainSet.keys(): # a user has first rate
        trainSet[user_in]={movie_in:rate_in}
        # Add new entry
        return userSim_full,trainSet
        
    # user_in have a new rating
    if movie_in not in trainSet[user_in]: # a user did not rated the movie
        trainSet[user_in]={movie_in:rate_in} # add the new rate to trainSet
        #calculate sim_factor for all users

        A=part1=part2=part3=part4=part5=part6=part7=part8=0
        for u in userSim_full[user_in]:
            e=0
            f=0
            g=0
            m = userSim_full[user_in][u][4]
            new_avg_a =  rate_in/(m+1) + userSim_full[user_in][u][5]*m/(m+1)
            dif_avg_a = (rate_in- userSim_full[user_in][u][5])/(m+1)
            A = userSim_full[user_in][u][0]
            #if u_y had not rated movie_i
            if user_in != u and movie_in not in trainSet[u]:
                
                f = m*dif_avg_a*dif_avg_a 
                - 2*dif_avg_a*(userSim_full[user_in][u][8]-m*userSim_full[user_in][u][5])
                
                e = -1*dif_avg_a*(userSim_full[user_in][u][7]- m*userSim_full[user_in][u][6])
                
                part1= userSim_full[user_in][u][1] + e
                part2= userSim_full[user_in][u][2] + f
                part3= userSim_full[user_in][u][3]
                part4= userSim_full[user_in][u][4] + 1
                part5= new_avg_a
                part6= userSim_full[user_in][u][6]  # same
                part7= userSim_full[user_in][u][7]
                part8= userSim_full[user_in][u][8]  # same
    
                if part2 == 0 or part3 == 0:
                    A1=0
                else:
                    A1 = part1 / (sqrt(part2) * sqrt(part3))              
                userSim_full[user_in][u] = (A,part1,part2,part3,part4,part5,part6,part7,part8)
  
            else:
                r_uy_ia = trainSet[u][movie_in]

                e= (rate_in - new_avg_a)*(r_uy_ia - userSim_full[user_in][u][7])
                - dif_avg_a*(userSim_full[user_in][u][7]- m*userSim_full[user_in][u][6])
              

                f = (rate_in - new_avg_a)*(rate_in - new_avg_a) + m*dif_avg_a*dif_avg_a 
                - 2*dif_avg_a*(userSim_full[user_in][u][8]-m*userSim_full[user_in][u][5])
                
                g= pow((r_uy_ia - userSim_full[user_in][u][6]),2)*1.0
                
                part1= userSim_full[user_in][u][1] + e
                part2= userSim_full[user_in][u][2] + f
                part3= userSim_full[user_in][u][3] + g
                part4= userSim_full[user_in][u][4] + 1
                part5= new_avg_a
                part6= userSim_full[user_in][u][6]  # same
                part7= userSim_full[user_in][u][7] + r_uy_ia
                part8= userSim_full[user_in][u][8] + rate_in  # same
                
                if part2 == 0 or part3 == 0:
                    A1=0
                else:
                    A1 = part1 / (sqrt(part2) * sqrt(part3))  
                userSim_full[user_in][u] = (A,part1,part2,part3,part4,part5,part6,part7,part8)

    return userSim_full,trainSet  
        #else 
        #Assumption: similarity table is recalibrate every day, So ->
        #Did not consider the situation that user update their rates for the same movie


# In[166]:

def getRecommendations_update(N,trainSet,userSim_full):
    pred = {}
    for user in trainSet.keys():    #generate new dic
        pred.setdefault(user,{})    #generate dic for users
        interacted_items = trainSet[user].keys() #get user rated movies
        average_u_rate = getAverageRating(user)  #average rates for user rated movies
        userSimSum = 0

#       sorted(test1['1'].items(),key = lambda x : x[1][0],reverse = True)[0:N]
        
        simUser = sorted(userSim_full[user].items(),key = lambda x : x[1][0],reverse = True)[0:N]
        
        for n, sim in simUser:  
            average_n_rate = getAverageRating(n)
            userSimSum += sim[0]   #calculate the similarity with neighbours
            for m, nrating in trainSet[n].items():  
                if m in interacted_items:  
                    continue  
                else:
                    pred[user].setdefault(m,0)
                    pred[user][m] += (sim[0] * (nrating - average_n_rate))
        for m in pred[user].keys():  
                pred[user][m] = average_u_rate + (pred[user][m]*1.0) / userSimSum
    return pred


# In[168]:

#format update_training_set to simulate the user new rate
def getTrainFormat(trainSet5K):
    new_rate_5k = []
    for user in trainSet5K:
        for movie in trainSet5K[user]:
            new_rate_5k.append([user,movie,trainSet5K[user][movie]])
    return new_rate_5k


# In[169]:

if __name__ == '__main__':
    print("loading data")
    trainSet,trainSet5K_1,trainSet5K_2,trainSet5K_3,testSet,movieUser,u2u = loadData()

    print("calibrate user similarity")
    userSim_full = getUserSim_full(u2u,trainSet)

    #format update_training_set_1
    new_rate_5k_1 = getTrainFormat(trainSet5K_1)

    #format update_training_set_2
    new_rate_5k_2 = getTrainFormat(trainSet5K_2)

    #format update_training_set_3
    new_rate_5k_3 = getTrainFormat(trainSet5K_3)

    start = time.clock()
    print("Start update similarity matrix 5K times")
    try:
        for u in new_rate_5k_1:
            userSim_full,trainSet= updateUserSim_full(u,userSim_full,trainSet)
    except:
        print("input error")
    end = time.clock()
    print("--Time of similarity update 50K times： %f s" % (end - start))
    
    
    start = time.clock()
    print("Start update similarity matrix 5K times")
    try:
        for u in new_rate_5k_2:
            userSim_full,trainSet= updateUserSim_full(u,userSim_full,trainSet)
    except:
        print("input error")
    end = time.clock()
    print("--Time of similarity update 2nd 50K times： %f s" % (end - start))
    
    
    start = time.clock()
    print("Start update similarity matrix 5K times")
    try:
        for u in new_rate_5k_3:
            userSim_full,trainSet= updateUserSim_full(u,userSim_full,trainSet)
    except:
        print("input error")
    end = time.clock()
    print("--Time of similarity update 3rd 50K times： %f s" % (end - start))

