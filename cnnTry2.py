import numpy as np
import pickle

phrases = pickle.load(open("phrases_train_100.p","r"))
senti = pickle.load(open("sentiThree_100.p","r"))
print "full loaded"
hidden_size = 1 # size of hidden layer of neurons
learning_rate = 1e-1
vocab_size = 100
filterN=9
classesN=3

Why = np.random.randn(classesN, filterN)*0.01 # hidden to output
by = np.zeros((classesN, 1)) # output bias

# model parameters
class Filter:
    def __init__(self,window):
        self.Wxh = np.random.randn(hidden_size, vocab_size*window)*0.01 # input to hidden
        #self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden      
        self.bh = np.zeros((hidden_size, 1)) # hidden bias       
        self.window=window
        self.activated = -1
        self.activatedValue = -1
        self.activatedInput = np.zeros((vocab_size*self.window,1))

    def convolutionL(self,inputP): # for each filter
        # ci = hs = f(Wxh . xs + bh)
        # C = [ c1, c2..] ; cs got from moving window
        #print "con"
        C=[]
        flag = False
        for i in xrange(len(inputP)-self.window+1):
            x = inputP[i:i+self.window]
            xn = np.zeros((vocab_size*self.window,1))
            m= []
            for a in x:
                m += [[i] for i in a]
            xn = np.copy(m)
            
            #xn = np.zeros((filterN,1))
            #xn = np.copy(x)
            #print xn,self.Wxh.shape,xn.shape,self.bh.shape
            #print xn.shape
            c = np.tanh(np.dot(self.Wxh, xn) + self.bh)
            C.append(c[0])
        else:
            C.append(np.zeros((1,)))
            flag = True
        self.activated = np.argmax(C)
        self.activatedValue = max(C)
        t = []
        #print "---",len(inputP)
        if  self.window>len(inputP)-self.activated:
            #global t
            #print "i : ",len(inputP), type(inputP)
            if self.window>len(inputP)-self.activated:#len(inputP)<self.window-self.activated:
                #global t
                t = inputP[:]
                inp = np.zeros((vocab_size,1))
                inpt = [0.0]*100
                inp = np.copy(inpt)
                t.append( inp*(self.window - len(inputP)))
                print "if t ", len(inputP),type(t) ,len(t), self.window, self.activated
        else:
            t = inputP[self.activated:self.activated+self.window]
            print "else t ", len(inputP), type(t) ,len(t), self.window, self.activated
        temp = []
        for a in t:
            temp += [[i] for i in a]
            #print "h"
        #print "t :",t[0]
        #print "a : ",a
        #print "m :",m
        self.activatedInput = np.copy(temp)
        #if self.window == 3:
            #print "------", self.window, self.activatedInput.shape
        #exit(0)
        #print "t : ",t, self.activatedInput.shape
        #print "C :",len(t[0]),len(m),C, self.activatedValue, self.activatedInput.shape,self.window
        return C

    def maxpoolL(self,inputP): # for each filter
        #c^ = max(C) ; C = [ c1, c2,...] ; cs got from convolution layer
        C = self.convolutionL(inputP)
        #print C, max(C), np.argmax(C)
        return max(C)

def forwardFeed(inputP, target):
    #print "ff"
    #print len(inputP)
    global count
    y,z = getFeaturesL(inputP)
    predicted = np.argmax(y)
    t = np.argmax(target)
    #print predicted, t
    if predicted == t:
        count += 1
    backwardFeed(y, target, z) 

def backwardFeed(y, target, z):
    #print "bf"
    #---- backpropagate softmax layer ----
    #E(3x1) = target - y
    #dWyh(3x9) = E(3x1) dot z.T(1x9)
    ntarget = np.zeros((classesN,1))
    ntarget = np.copy([[i] for i in target])
    #print y, y.shape,target,ntarget, ntarget.shape
    E = np.subtract(ntarget, y)
    #print "E :",ntarget, y, E.shape
    dWhy = np.dot(E, z.T)
    #print dWhy.shape, E.shape, z.T.shape
    dby = E 

    #---- backpropagate convolution ----
    ## Go through all filters and propagate from inError to each filter
    # outError(9x1) = Why.T(9x3) dot E(3x1)
    outError = np.dot(Why.T, E)
    #print inError.shape, Why.T.shape, E.shape

    for i in xrange(len(filterSet)):
        # localError(1x1) = 1 - filterSet[i].activatedValu**2
        # inError(1x1) = outError[i](1x1)
        # filterSet[i].dWxh(1x200) = inError(1x1) dot localError(1x1) dot activatedInput.T(1x200) 
        localError = 1 - filterSet[i].activatedValue**2
        inError = outError[i]
        dWxh = np.dot(np.dot(inError, localError), filterSet[i].activatedInput.T)
        print dWxh.shape
        dbh = inError
        print filterSet[i].window,filterSet[i].Wxh.shape,dWxh.shape
        for param, dparam in zip([filterSet[i].Wxh, filterSet[i].bh],
                                [dWxh, dbh]):
            param += -learning_rate * dparam

    for param, dparam in zip([Why, by],
                                [dWhy, dby]):
        param += -learning_rate * dparam
    
        
def getFeaturesL(inputP): # after maxpool of all filters, one element per filter
    # z
    # call convolutionL() again and again
    #print "getFeatures"
    z = []
    for f in filterSet:
        #f = filterSet[i]#Filter(1,inputP)  
        z.append(f.maxpoolL(inputP))
    #print "z : ",z, len(z)
    return softmaxL(z)
    

def softmaxL(z):
    #print "softmax"
    # y = softmax(Why . z + by )
    zn = np.zeros((filterN,1))
    zn = np.copy(z)
    #print z,zn,Why.shape,zn.shape,by.shape
    p = np.dot(Why, zn) + by
    #print "pppp",p
    y = np.exp(p) / np.sum(np.exp(p))
    #print "y : ",type(y), y.shape
    return y,zn

filterSet = []
for i in xrange(filterN/3):
    filterSet.append(Filter(1))
    filterSet.append(Filter(2))
    filterSet.append(Filter(3))
    
count = 0
l = 10
def main():
    n = 0    
    while n<l:
        target = [0]*classesN
        target[senti[n]]=1
        forwardFeed(phrases[n],target)
        n+=1
    print "main"

    print "Accuracy : ",count/float(l)
if __name__ == '__main__':
    main()
