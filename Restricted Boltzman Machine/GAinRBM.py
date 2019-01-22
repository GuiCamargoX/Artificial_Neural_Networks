import numpy as np
import random as rd
from sklearn.neural_network import BernoulliRBM
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import RepeatedKFold
import statistics

mnist = input_data.read_data_sets('MNIST_data', one_hot = False)
X = np.vstack((mnist.train.images,mnist.test.images))
X = X>0.5

# Calculating Decode
def Decode_X4(chrom):
    t = 1
    X0_num_Sum = 0
    
    for i in range(len(chrom)):
        X0_num = chrom[-t]*(2**i)
        X0_num_Sum += X0_num
        t = t+1
    
    return (X0_num_Sum * Precision_X4) + a_X4

def Fight(Troops):
    AC = []
    
    for Warrior in Troops:
        
        AC.append( CalculateObjectFunction(Warrior) )
    
    m = AC.index(max(AC))
    
    return Troops[m]

def CalculateObjectFunction(chrom):
    W_Comb_1 = int(chrom[0])
    W_Comb_2 = int(chrom[1])
    W_Comb_3 = int(chrom[2])
    Decoded_X4_W = Decode_X4(chrom[3:])
                
    kf = RepeatedKFold(n_splits=5, n_repeats= 2)
    AC = []

    for train, test in kf.split(X):
        model = BernoulliRBM(n_components=W_Comb_1, learning_rate= Decoded_X4_W, batch_size=W_Comb_2
                             , n_iter= W_Comb_3, verbose=1, random_state=0)
        model.fit(X[train])
        AC.append(  model.score_samples(X[test]).mean() )
        
    return statistics.mean( AC )
    

def Mute(Child, index, UB, LB):
    gene_Mut = Child[index]
    Ran_M_1 = np.random.rand()
    
    if Child[index] == UB or Child[index] == LB:#bound
        return gene_Mut
    
    if Ran_M_1 < p_m_comb:
        Ran_M_2 = np.random.rand()
        if Ran_M_2 >= 0.5 :
            gene_Mut = Child[index] + 2
        else:
            gene_Mut = Child[index] - 2

    return gene_Mut

def MuteContinuous(Child):
    
    for i in range(len(Child)):
        
        Ran_Mut = np.random.rand() # Probablity to Mutate
            
        if Ran_Mut < p_m_con: # If probablity to mutate is less than p_m, then mutate
                
            if Child[i] == 0:
                Child[i] = 1
            else:
                Child[i] = 0
            
            Mutated_Child = Child
                
        else:
            Mutated_Child = Child
    
    return Mutated_Child


### The solver has no crossover because mutation is enough, since it only has two values
### VARIABLES ###
### VARIABLES ###
p_c_con = 1 # Probability of crossover
p_c_comb = 0.3 # Probability of crossover for integers
p_m_con = 0.1 # Probability of mutation
p_m_comb = 0.2 # Probability of mutation for integers
pop = 30 # Population per generation
gen = 30  # Number of generations
### VARIABLES ###
### VARIABLES ###


### Combinatorial ###
UB_X1 = 256 # X1, Number of Components(hidden Neurons)
LB_X1 = 256
UB_X2 = 10# X2, batch_size( Number of examples per Maxbatch) 
LB_X2 = 10
UB_X3 = 10 # X3, n_inter( number of iter over the training dataset)
LB_X3 = 10 
# X4 = Learning Rate real number
a_X4 = 0.01 # Lower bound of X
b_X4 = 0.3 # Upper bound of X
      # Where the 15 bits represent X4
XY0 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1]) # Initial solution
l_X4 = len(XY0) # Length of Chrom. X
Precision_X4 = (b_X4 - a_X4)/((2**l_X4)-1) # precision

### Continuous ###
Init_Sol = XY0.copy()

n_list = np.empty((0,len(XY0)+3))
Sol_Here = np.empty((0,len(XY0)+3))


#inic population
for i in range(pop): # Shuffles the elements in the vector n times and stores them
    X1 = rd.randrange(LB_X1,UB_X1+1,2)
    X2 = rd.randrange(LB_X2,UB_X2+1,2)
    X3 = rd.randrange(LB_X3,UB_X3+1,2)
    rd.shuffle(XY0)
    Sol_Here = np.append((X1,X2,X3),XY0)
    n_list = np.vstack((n_list,Sol_Here))



One_Final_Guy = np.empty((0,len(Sol_Here)+2))
One_Final_Guy_Final = []

Max_for_Generation_X = np.empty((0,len(Sol_Here)+2))
Max_for_all_Generations_for_Mut = np.empty((0,len(Sol_Here)+2))


for i in range(gen):
    
    
    New_Population = np.empty((0,len(Sol_Here))) # Saving the new generation
    
    X_1 = np.empty((0,len(Sol_Here)+1))
    X_2 = np.empty((0,len(Sol_Here)+1))
    
    Max_in_Generation_X_1 = []
    Max_in_Generation_X_2 = []
    
    
    All_in_Generation_X = np.empty((0,len(Sol_Here)+1))
    Final_Best_in_Generation_X = []
    Worst_Best_in_Generation_X = []
    
    
    print()
    print("--> GENERATION: #",i)
    
    Family = 1

    for j in range(int(pop/2)): # range(int(pop/2))
            
        print()
        print("--> FAMILY: #",Family)
              
            
        # Tournament Selection
        # Tournament Selection
        # Tournament Selection
        
        Parents = np.empty((0,len(Sol_Here)))
        
        for i in range(2):
            
            Battle_Troops = []
            
            Warrior_1_index = np.random.randint(0,len(n_list)) #3
            Warrior_2_index = np.random.randint(0,len(n_list)) #5
            Warrior_3_index = np.random.randint(0,len(n_list))
            
            while Warrior_1_index == Warrior_2_index:
                Warrior_1_index = np.random.randint(0,len(n_list))
            while Warrior_2_index == Warrior_3_index:
                    Warrior_3_index = np.random.randint(0,len(n_list))
            while Warrior_1_index == Warrior_3_index:
                    Warrior_3_index = np.random.randint(0,len(n_list))
            
            Warrior_1 = n_list[Warrior_1_index]
            Warrior_2 = n_list[Warrior_2_index]
            Warrior_3 = n_list[Warrior_3_index]
            
            
            Battle_Troops = [Warrior_1,Warrior_2,Warrior_3]
            
            Winner = Fight(Battle_Troops)
            
            Parents = np.vstack((Parents,Winner))
        
        print()
        print("Parents:",Parents)
        
        
        Parent_1 = Parents[0]
        Parent_2 = Parents[1]
        
        
        # Crossover
        # Crossover
        # Crossover
        
        
        Child_1 = np.empty((0,len(Sol_Here)))
        Child_2 = np.empty((0,len(Sol_Here)))
        
        
        # Crossover the Integers
        # Combinatorial
        
        # For X1
        # For X1
        Ran_CO_1 = np.random.rand()
        if Ran_CO_1 < p_c_comb:
            # For X1
            Int_X1_1 = Parent_2[0]
            Int_X1_2 = Parent_1[0]
        else:
            # For X1
            Int_X1_1 = Parent_1[0]
            Int_X1_2 = Parent_2[0]
        
        # For X2
        # For X2
        Ran_CO_1 = np.random.rand()
        if Ran_CO_1 < p_c_comb:
            # For X2
            Int_X2_1 = Parent_2[1]
            Int_X2_2 = Parent_1[1]
        else:
            # For X2
            Int_X2_1 = Parent_1[1]
            Int_X2_2 = Parent_2[1]

        # For X3
        # For X3
        Ran_CO_1 = np.random.rand()
        if Ran_CO_1 < p_c_comb:
            # For X3
            Int_X3_1 = Parent_2[2]
            Int_X3_2 = Parent_1[2]
        else:
            # For X3
            Int_X3_1 = Parent_1[2]
            Int_X3_2 = Parent_2[2]
        
        
        # Continuous
        # Where to crossover
        # Two-point crossover
        
        Ran_CO_1 = np.random.rand()
        
        if Ran_CO_1 < p_c_con:
        
            Cr_1 = np.random.randint(3,len(Sol_Here))
            Cr_2 = np.random.randint(3,len(Sol_Here))
                
            while Cr_1 == Cr_2:
                Cr_2 = np.random.randint(3,len(Sol_Here))
            
            if Cr_1 > Cr_2:
                Cr_1, Cr_2 = Cr_2, Cr_1
                
            Cr_2 = Cr_2 + 1
                
            Copy_1 = Parent_1[3:]
            Mid_Seg_1 = Parent_1[Cr_1:Cr_2]
                
            Copy_2 = Parent_2[3:]
            Mid_Seg_2 = Parent_2[Cr_1:Cr_2]
                
            First_Seg_1 = Parent_1[3:Cr_1]
            Second_Seg_1 = Parent_1[Cr_2:]
                
            First_Seg_2 = Parent_2[3:Cr_1]
            Second_Seg_2 = Parent_2[Cr_2:]
                
            Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Second_Seg_1))
            Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Second_Seg_2))
                
            Child_1 = np.insert(Child_1,0,(Int_X1_1,Int_X2_1, Int_X3_1))###
            Child_2 = np.insert(Child_2,0,(Int_X1_2,Int_X2_2, Int_X3_2))
            
        else:
            
            Child_1 = Parent_1[3:]
            Child_2 = Parent_2[3:]

            Child_1 = np.insert(Child_1,0,(Int_X1_1,Int_X2_1, Int_X3_1))###
            Child_2 = np.insert(Child_2,0,(Int_X1_2,Int_X2_2, Int_X3_2))

        
        Mutated_Child_1 = []
        Mutated_Child_2 = []
        # Combinatorial
        
        # For X1
        C_X1_M1 = Mute(Child_1,0, UB_X1, LB_X1)
        # For x2
        C_X2_M1 = Mute(Child_1,1, UB_X2, LB_X2)
        # For x3
        C_X3_M1 = Mute(Child_1,2, UB_X3, LB_X3)
           
        # Continuous
        Mutated_Child_1 = MuteContinuous(Child_1[3:])

        Mutated_Child_1 = np.insert(Mutated_Child_1,0,(C_X1_M1,C_X2_M1,C_X3_M1))
        
    
        # For X1
        C_X1_M2 = Mute(Child_2,0, UB_X1, LB_X1)
        # For x2
        C_X2_M2 = Mute(Child_2,1, UB_X2, LB_X2)
        # For x3
        C_X3_M2 = Mute(Child_2,2, UB_X3, LB_X3)
           
        # Continuous
        Mutated_Child_2 = MuteContinuous(Child_2[3:])

        Mutated_Child_2 = np.insert(Mutated_Child_2,0,(C_X1_M2,C_X2_M2,C_X3_M2))

        # Calculate fitness values of mutated children
        OF_So_Far_MC_1 = CalculateObjectFunction(Mutated_Child_1)
        OF_So_Far_MC_2 = CalculateObjectFunction(Mutated_Child_2)
        
        X_1 = np.column_stack((OF_So_Far_MC_1, Mutated_Child_1[np.newaxis] ))
        X_2 = np.column_stack((OF_So_Far_MC_2, Mutated_Child_2[np.newaxis] ))
        
        All_in_Generation_X = np.vstack((All_in_Generation_X, X_1, X_2))
        
        New_Population = np.vstack((New_Population,Mutated_Child_1,Mutated_Child_2))
        Family = Family+1
    
    #FindBestOfGeneration
    best = np.where(All_in_Generation_X[:,0] == np.max(All_in_Generation_X[:,0]))
    Final_Best_in_Generation_X = All_in_Generation_X[best[0][0]]
    #FindWorstOfGeneration
    worst = np.where(All_in_Generation_X[:,0] == np.max(All_in_Generation_X[:,0]))
    Worst_Best_in_Generation_X = All_in_Generation_X[worst[0][0]]  
    
    Darwin_Guy = Final_Best_in_Generation_X[1:].tolist()
    Not_So_Darwin_Guy = Worst_Best_in_Generation_X[1:].tolist()
    
    #search in population
    Best_1 = np.where((New_Population == Darwin_Guy).all(axis=1))
    Worst_1 = np.where((New_Population == Not_So_Darwin_Guy).all(axis=1))
    #elitism
    New_Population[Worst_1] = Darwin_Guy
    
    n_list = New_Population
    
    #save
    Max_for_Generation_X = np.insert(Final_Best_in_Generation_X, 0, i)
    Max_for_all_Generations_for_Mut = np.vstack( (Max_for_all_Generations_for_Mut, Max_for_Generation_X) )
    
one = np.where(Max_for_all_Generations_for_Mut[:,1] == max(Max_for_all_Generations_for_Mut[:,1]) )
One_Final_Guy_Final = Max_for_all_Generations_for_Mut[one[0][0]]

print()
print()
print("The High Accuracy is:",(One_Final_Guy_Final[1]))
print("Number of Components(hidden Neurons):",One_Final_Guy_Final[2])
print("batch_size:",One_Final_Guy_Final[3])
print("n_inter:",One_Final_Guy_Final[4])
print("Learning Rate:", Decode_X4(One_Final_Guy_Final[5:]))
