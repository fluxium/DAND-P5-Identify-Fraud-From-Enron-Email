def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    
    ### your code goes here!
    from time import time
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    
    reg.fit(ages_train, net_worths_train)
    
    t_fit = time()
    reg.fit(ages_train, net_worths_train)
    print "training time:", round(time()-t_fit, 3), "s"
    
    return reg