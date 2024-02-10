from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.widgets as wg




def iris_data_set():
    """First of all, we set some fuctions to obtain dataset, such as two iris data sets. 
    The first data set has two kinds of flowers which ovbiously can be categoraized by this algorithm. 
    It is called 'easy' data set on this project since it takes a few times to be categorized by this algorithm."""
    iris = load_iris()
    Y =iris.target[0:100]
    for i in range(len(Y)): #setosa:1, versicolor:-1
        if Y[i] == 0:
            Y[i] = 1
        elif Y[i] == 1:
            Y[i] = -1

    X = iris.data[0:100, [0, 2]]
    return X, Y


def iris_data_set2(): #another hard iris dataset
    """At this fucntion, we obtain different iris dataset.
      It is called 'hard' dataset since this algorithm cannnot categorize them. 
      In addition, it needs many trainning to get the line close to between those groups."""
    iris = load_iris()
    Y =iris.target[50:150]
    for i in range(len(Y)): #setosa:1, versicolor:-1
        if Y[i] == 0:
            Y[i] = 1
        elif Y[i] == 1:
            Y[i] = -1

    X = iris.data[50:150, [0, 2]]
    return X, Y


def randomdataset(): # to make random dataset
    """At this function, we obtain random dataset.
    By using numpy library, we obtain random integers, which range is 0 to 10."""
    rng = np.random.default_rng(0)
    ran_x_set = rng.random((100, 2))
    ran_y_set = [1]*50 + [-1]*50
    
    for i in range(50):
       
        ran_x_set[i][0] = round(ran_x_set[i][0], 2) + rng.integers(10)
        ran_x_set[i][1] = round(ran_x_set[i][1], 2) + rng.integers(10)
       
    for i in range(50, 100):

        ran_x_set[i][0]=round(ran_x_set[i][0], 2) + rng.integers(10)
        ran_x_set[i][1] = round(ran_x_set[i][1], 2) + rng.integers(10)

    return ran_x_set, ran_y_set

class Perceptron:

    """Percertron is an algorithm for supervised learning of single layer binary linear classifiers. 
    At this program, we see two different groups on the plot. 
    Perceptron Algorithm tries to classify those by the linear line. 
    There are three different datasets as I mentioned above. 
    The first dataset called 'easy' can be classfied in a few times. 
    The second one called 'hard' cannnot be classfied, and it needs many tranning the linear line goes to the center of two groups. 
    The third dataset called 'random' ovbiously cannot be classified. """

    def __init__(self, n_inters, learning_rate=0.1, random_state=1):

        """At this class, we make perceptron. At the beginning, we set weights, which is None, and n_inters, which is to count how many trainning it does, 
        and random_state, which is to initilize the weights."""
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.n_inters = n_inters # how many traing
         # to initialize weights
        
    def fit(self, X, Y):

        """At this function, we mainly update weights of datasets. 
        First of all, we initialize the weights with small random numbers in a normal distribution with standard deviation(0.01).
        After that, we update the weights by trainning. Before the update, we check if we need to upadate weights at line 86."""
        if self.n_inters == 1:
            ranum = np.random.RandomState(self.random_state)
            #to make random number corresponding to normal distribution
            self.weights = ranum.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.n_inters +=1
        self.errors = []
        for _ in range(self.n_inters):
            
            error = 0
            for i, target in zip(X, Y):
                
                update = self.learning_rate * (target - self.predict(i))
                # if (target - self.predict(i)) is 0, which means true, no do update, else, do update
                self.weights[1:] += update*i #update weights from 1
              
                self.weights[0] += update #update weights at 0 (bias)
        
                # count how many errors
                error += int(update != 0.0) # when cant predict
            
            self.errors.append(error)
        
        return self

    def inf_input(self,X): #step function 内積

        """At this function, we multipy all input values with corresponding weight values and then add to calculate the weighted sum."""

        return np.dot(X, self.weights[1:]) + self.weights[0]   #inner product


    def predict(self,X):

        """At this function, we desitingish between -1 and 1 based on the value from def inf_input.
        If the value is bigger or equal to 0, return 1, else, return -1."""

        # to desitingish between -1 and 1
        return np.where(self.inf_input(X) >= 0.0, 1, -1) # if it is true, return 1, else, return -1
#data extracted
 
def plot_decision_regions(X, Y, classifier, resol=0.02):

    """At this fucntion, we obtain datasets, and value after trainning. First, we make the grid of the map, which is up to minimum and maximum values of datasets.
    After that, we resize according to the result of predict.
    In addition, we see the plot of datasets with the line of perceptron."""

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Y))])  #sort list and delete unnecessay info, result:[red, blue]
    
    x1_min, x1_max = X[:, 0].min() -2.5, X[:, 0].max() + 2.5 #setosa
    
    x2_min, x2_max = X[:, 1].min() -2.5, X[:, 1].max() + 2.5 #versicolor
    
    # to make grid
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resol), np.arange(x2_min, x2_max, resol))
    
    result = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # to make simple dimension
   
    result = result.reshape(xx1.shape) #resize according to the result of predict

    plt.contourf(xx1, xx2, result, alpha=0.3, cmap=cmap)  # to make plot of contourf
  
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    
    for i, cl in enumerate(np.unique(Y)): #plot each class of training data. 
        plt.scatter(x=X[Y == cl, 0], 
                    y=X[Y == cl, 1],
                    alpha=0.8, 
                    c=colors[i],
                    marker=markers[i], 
                    label=cl, 
                    edgecolor='black')
        
"""'n' is to count how many trainning this algorithm does."""
n = 1

"""'count' is to check which dataset this algorithm is trainning now."""
count = 0
easy_p = Perceptron(n)
hard_p = Perceptron(n)
random_p =  Perceptron(n)
"At the beginning, we set the percetron for each data. "
X, Y = iris_data_set() # at the beginning, we show the easy iris dataset. 


def btn_exit(event): 

    """At this fucntion, there is a botton at right-upper to finish this program."""

    exit()

def btn_train(evnt): # when we train data. 

    """After we push the botton called 'trainning' on the plot, this code comes here. we check which dataset we need to do trainning.
    After that, the dataset goes to perceptron, then we can see the result of trainning at Function plot_decision_regions. 
    There are the limit of tranning. we can do trainning about 90 times. After that, the issue comes up, which is recurison excess. 
    Thus, at 90 times tranning, we see the sentence, which implies we have to finish it.
    In addition, we can change the datasets to do trainning by  pushing the bottons called 'easy', 'hard', or 'random'."""

    global count, easy_p, hard_p, random_p
    if count == 1:
        X, Y = randomdataset()
        ppn = random_p
    elif count == 2:
        X, Y = iris_data_set2()
        ppn = hard_p
    else:
        X, Y = iris_data_set()
        ppn = easy_p
    
    
    

    ppn.fit(X, Y)
    plt.figure()
    
    plot_decision_regions(X, Y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    
    if n <= 90: #about 90 times are the maximam recursion.
        ax = plt.axes([0.03, 0.03, 0.15, 0.05])
        #training botton
        btn2 = wg.Button(ax, '$Training$', color='yellow', hovercolor='0.98')
        btn2.on_clicked(btn_train)
        ax = plt.axes([0.8, 0.9, 0.15, 0.05])
        #finish botton
        btn3 = wg.Button(ax, '$Finish$', color='white', hovercolor='0.98')
        btn3.on_clicked(btn_exit)

        ax = plt.axes([0.8, 0.03, 0.15, 0.05])
        btn5 = wg.Button(ax, '$random$', color='grey', hovercolor='0.98')
        btn5.on_clicked(btn_random)
        ax = plt.axes([0.425, 0.03, 0.15, 0.05])

        if count == 2: # if we are taining hard iris dataset. 
            btn4 = wg.Button(ax, '$easy$', color='orange', hovercolor='0.98')
            btn4.on_clicked(btn_change)
            # if we are taining easy iris dataset. 
            
        else: 
            btn4 = wg.Button(ax, '$hard$', color='green', hovercolor='0.98')
            btn4.on_clicked(btn_change)
        
        

    else: # we have to finish it.
        
        plt.text(2, 2, "You cannot train data.", size =20)
        ax = plt.axes([0.8, 0.03, 0.15, 0.05])
        btn3 = wg.Button(ax, '$Finish$', color='white', hovercolor='0.98')
        btn3.on_clicked(btn_exit)
    plt.show()

    
def btn_random(event):

    """At this fucntion, after pushing the botton called random, the code comes here. 
    we obtain the dataset of random and go to the function see_Scatter_plot to see scatter plot of random dataset."""

    global count
    
    count = 1
   
    plt.figure()
    X, Y = randomdataset()
    see_Scatter_plot(X)

def btn_change(event):

    """At this function, we change the dataset to do tranning. 
    If we push 'easy' on the plot, we obtan 'easy' dataset. If we push 'hard', we obtan 'hard' dataset.
    And we see scatter plot. """

    global count
    
    if count == 2: # if count == 2, see scatter graph of the easy iris dataset .
        count = 0
        
        plt.figure()
        X, Y = iris_data_set()
        see_Scatter_plot(X)
    else: # if count == 2, see scatter graph of the hard iris dataset
        count = 2
       
        plt.figure()
        X, Y = iris_data_set2()
        see_Scatter_plot(X)
    

def see_Scatter_plot(X):

    """ When we start to run this code, first of all, we see scatter plot of the easy dataset. 
    Also, if we push some bottons which represents dataset. This code comes here to show other scatter plot of other datasets.
    In addition, there are some bottons including 'Tranning', 'easy' or 'hard', and 'random'. """
    
    global count
    
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    # versicolor, ✕ is blue
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    
    ax1 = plt.axes([0.05, 0.05, 0.15, 0.05])
    btn1 = wg.Button(ax1, '$Training$', color='yellow', hovercolor='0.98')
    btn1.on_clicked(btn_train)

    ax = plt.axes([0.8, 0.03, 0.15, 0.05])
    btn3 = wg.Button(ax, '$random$', color='grey', hovercolor='0.98')
    btn3.on_clicked(btn_random)
    ax = plt.axes([0.425, 0.03, 0.15, 0.05])

    if count == 2:
        btn4 = wg.Button(ax, '$easy$', color='orange', hovercolor='0.98')
        btn4.on_clicked(btn_change)

    else:
        btn4 = wg.Button(ax, '$hard$', color='green', hovercolor='0.98')
        btn4.on_clicked(btn_change)
    plt.show()





see_Scatter_plot(X)

