import pandas
import numpy as np
import numpy.linalg
import math
import matplotlib.pyplot as plt


count = 500
count_float = 500.0

#Plotting function, takes points and dimensions to plot
def plot(x11,x12,x21,x22, dim1,dim2):
    # Getting points in proportions for plotting
    #finding minimum and maximum from each array
    minX11 = min(x11)
    maxX11 = max(x11)
    minX12 = min(x12)
    maxX12 = max(x12)
    propX11 = maxX11 - minX11
    propX12 = maxX12 - minX12

    minX21 = min(x21)
    maxX21 = max(x21)
    minX22 = min(x22)
    maxX22 = max(x22)
    propX21 = maxX21 - minX21
    propX22 = maxX22 - minX22

    #divide the array from the difference of minimum and maximum
    x11 = [x / propX11 for x in x11]
    x12 = [x / propX12 for x in x12]
    x21 = [x / propX21 for x in x21]
    x22 = [x / propX22 for x in x22]

    plt.scatter( x11 , x12, marker = "o" )
    plt.scatter( x21 , x22, marker = "*" )

    #plt.axis([-1.5, 1.5, -1.5, 1.5])
    #plt.hlines(0, -1.5, 1.5)
    plt.xlabel(dim1)
    #plt.vlines(0, -1.5, 1.5)
    plt.ylabel(dim2)
    plt.show()

#Calcualate sample mean and cov for one partcular feature using Parzen window
def uni_var_parzen(X, mean, sigma):
    fx_values = []
    N = []
    summ = 0
    for i in range(0,count-1):
        exp_term = (X[i] - mean ) * ( X[i] -mean)
        if sigma == 0:
            exp_term = 1
            sigma = 0.01
        else:
            exp_term = np.exp(-(exp_term / (2 * sigma * sigma)))
        fx = (1/ (sigma*np.sqrt((2*3.14)) )) * exp_term
        fx = np.float64(fx)
        fx_values.append(fx)
        N.append(i)
        summ = summ + fx

    #Sample mean and variance
    sample_mean = (summ)/count
    sum1 = 0
    for i in range(0,count-1):
        #for variance calc
        diff = (X[i] - sample_mean)*(X[i] - sample_mean)
        sum1 = sum1 + diff

    sample_var = sum1 / (count-1)
    # Plot the curves
    #plt.plot(N, fx_values , color='black', label='')
    #plt.xlabel("N")
    #plt.ylabel("Fx values")
    #plt.show()
    return sample_mean, sample_var

#Plotting function, takes points and dimensions to plot
def plot_desc(x11,x12,x21,x22, dim1,dim2,curve_x1,curve_x21,curve_x22):
    # Getting points in proportions for plotting
    #finding minimum and maximum from each array
    minX11 = min(x11)
    maxX11 = max(x11)
    minX12 = min(x12)
    maxX12 = max(x12)
    propX11 = maxX11 - minX11
    propX12 = maxX12 - minX12

    minX21 = min(x21)
    maxX21 = max(x21)
    minX22 = min(x22)
    maxX22 = max(x22)
    propX21 = maxX21 - minX21
    propX22 = maxX22 - minX22

    prop_curve_x1 = ( max(curve_x1) - min(curve_x1))
    prop_curve_x21 = ( max(curve_x21) + min(curve_x21))
    prop_curve_x22 = ( max(curve_x22) + min(curve_x22))

    #divide the array from the difference of minimum and maximum
    x11 = [x / propX11 for x in x11]
    x12 = [x / propX12 for x in x12]
    x21 = [x / propX21 for x in x21]
    x22 = [x / propX22 for x in x22]
    curve_x1 = [ x*2 / prop_curve_x1 for x in curve_x1]
    curve_x21 = [ x / prop_curve_x21 for x in curve_x21]
    curve_x22 = [ x / prop_curve_x22 for x in curve_x22]

    print("curvex1")
    print(curve_x1)
    print("x21")
    print(curve_x21)
    print("x22")
    print(curve_x22)

    plt.scatter( x11 , x12, marker = "o" )
    plt.scatter( x21 , x22, marker = "*" )
    # Plot the curves
    plt.plot(curve_x1, curve_x21, color = 'black', label = 'x2 first root' )
    plt.plot(curve_x1, curve_x22, color = 'black', label = 'x2 second root' )

    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.hlines(0, -1.5, 1.5)
    plt.xlabel(dim1)
    plt.vlines(0, -1.5, 1.5)
    plt.ylabel(dim2)
    plt.show()


#this function calculates Quadratic classifier
def calc_bayes(M1, M2, cov1, cov2, x11, x12, x13, x21, x22, x23 ):
    # Ax, Bx, Cx are the values of A,B,C in X domain
    #print("np.linalg.inv(cov2)")
    #print(np.linalg.inv(cov2))
    #print("np.linalg.inv(cov1)")
    #print(np.linalg.inv(cov1))

    Ax = (np.linalg.inv(cov2) - np.linalg.inv(cov1)) / 2
    Bx = ((np.matrix.transpose(M1) * np.linalg.inv(cov1)) - (np.matrix.transpose(M2) * np.linalg.inv(cov2)))
    # As apriori probabilities P1 and P2 are equal because of distribution, so term log P1/P2 would be zero
    if np.linalg.det(cov1) <=0 or np.linalg.det(cov2) <=0:
        Cx = 0
    else:
        Cx = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))

    # quadratic equation for X1 - X2, and solving it for x2 while feeding the values of x1
    # Getting the values of x2 by the formula ( -b +/- srt (b*b - 4*a*c) ) / (2*a)
    a = Ax.item(1, 1)
    x2_root1 = []
    x2_root2 = []
    x12_values = []
    # feeding values of i for x1
    for i in range(-10, 15, 1):
        b = (Ax.item(1, 0) * i) + (Ax.item(0, 1) * i) + Bx.item(1)
        c = (Ax.item(0, 0) * i * i) + (Bx.item(0) * i) + Cx
        if ((b * b - 4 * a * c) > 0):
            x12_values.append(i)
            x2_root1.append((- b + np.sqrt(b * b - 4 * a * c)) / (2 * a))
            x2_root2.append((- b - np.sqrt(b * b - 4 * a * c)) / (2 * a))
        else:
            continue

    # quadratic equation for X1 - X3, and solving it for x3 while feeding the values of x1
    # Getting the values of x3 by the formula ( -b +/- srt (b*b - 4*a*c) ) / (2*a)
    a = Ax.item(2, 2)
    x3_root1 = []
    x3_root2 = []
    x13_values = []
    # feeding values of i for x1
    for i in range(-10, 15, 1):
        b = (Ax.item(2, 0) * i) + (Ax.item(0, 2) * i) + Bx.item(2)
        c = (Ax.item(0, 0) * i * i) + (Bx.item(0) * i) + Cx

        if ((b * b - 4 * a * c) > 0):
            x13_values.append(i)
            x3_root1.append((- b + np.sqrt(b * b - 4 * a * c)) / (2 * a))
            x3_root2.append((- b - np.sqrt(b * b - 4 * a * c)) / (2 * a))
        else:
            continue

    # Plot points and Classification Curve for X in x1-x2 and x1-x3 dimensions
    plot_desc(x11, x12, x21, x22, 'X1', 'X2', x12_values, x2_root1, x2_root2)
    plot_desc(x11, x13, x21, x23, 'X1', 'X3', x13_values, x3_root1, x3_root2)

#this function claculates classification accuracy
def class_accuracy(M1, M2, cov1,cov2,x11,x12,x13,x14,x15,x16,x21,x22,x23,x24,x25,x26):

    # Ax, Bx, Cx are the values of A,B,C in X domain
    Ax = (np.linalg.inv(cov2) - np.linalg.inv(cov1)) / 2
    Bx = ((np.matrix.transpose(M1) * np.linalg.inv(cov1)) - (np.matrix.transpose(M2) * np.linalg.inv(cov2)))
    # As apriori probabilities P1 and P2 are equal because of distribution, so term log P1/P2 would be zero
    Cx = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))

    # Test and Build a decesion matrix
    # Decision matrix first element true- true for class w1, and fourth element is true-ture for w2

    True_count_Xw1w1 = 0
    True_count_Xw2w2 = 0

    #running loop 5 times for 5X validation
    for j in range(0, 4):
        for i in range(100*j , 100*j+100):
            Xx1 = np.matrix([[x11[i]], [x12[i]], [x13[i]], [x14[i]], [x15[i]], [x16[i]] ])
            fn = ((np.matrix.transpose(Xx1) * Ax) * Xx1) + ((Bx) * Xx1) + Cx

            # check the value of expression for X, if it is Greater than 0, point X belongs to class w1,
            # else it belongs to w2
            if fn > 0:
                True_count_Xw1w1 = True_count_Xw1w1 + 1
            else:
                True_count_Xw2w2 = True_count_Xw2w2 + 1

            Xx2 = np.matrix([[x21[i]], [x22[i]], [x23[i]], [x24[i]], [x25[i]], [x26[i]] ])
            fn = ((np.matrix.transpose(Xx2) * Ax) * Xx2) + (Bx * Xx2) + Cx

            if fn > 0:
                True_count_Xw1w1 = True_count_Xw1w1 + 1
            else:
                True_count_Xw2w2 = True_count_Xw2w2 + 1

    # calculate accuracy
    accuracy = ((True_count_Xw1w1 + True_count_Xw2w2) / 1000) * 100

    print("c. The accuracy : ")
    print(accuracy)

def divide_2d_matrix_by_n(m):
    #print("printing matrix in divide function")
    for i in range(0,3,1):
        #print("printing "+str(m[i]))
        m[i] = [x / count for x in m[i]]
        #print("after divide "+str(m[i]))

    return m

#Find covariances between two variable x and y
def find_cov(x,y,Mx, My):
    sum = 0
    for i in range(0,count-1):
        #print("i = "+str(i))
        #print("Xi = "+str(x[i]))
        sum = sum + ((x[i] - Mx)*(y[i] - My))
    sum = sum / (count_float -1)
    return sum


## Fisher's Descriminant
def fisher_desc( M1, M2, Sigma1, Sigma2,x11,x21 ):

    Sw = Sigma1 + Sigma2
    W = np.linalg.inv(Sw) * (M1 - M2)
    M1_plane = np.matrix.transpose(W)* M1
    M2_plane = np.matrix.transpose(W)* M2
    Sigma1_plane = np.matrix.transpose(W)*Sigma1*W
    Sigma2_plane = np.matrix.transpose(W) * Sigma2 * W

    #print("M1 and M2 plane = "+str(M1_plane)+" and "+str(M2_plane))
    #print("Sigma1 and SIgma 2 plane = "+str(Sigma1_plane)+" and "+str(Sigma2_plane))

    #Using CLassifier for Unidimensional
    #A*x^2 + B*x + C > w1 or < w2
    A = ( pow(Sigma1_plane,2) - pow(Sigma2_plane,2) )
    B = ( 2 * M1_plane * pow(Sigma2_plane,2) ) - ( 2 * M2_plane * pow(Sigma1_plane,2) )
    C = ( pow(M2_plane,2) * pow(Sigma1_plane,2) ) - ( pow(M1_plane,2) * pow(Sigma2_plane,2) ) - ( 2* pow(Sigma2_plane,2)*pow(Sigma1_plane,2) )

    root1 = (- B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    root2 = (- B - np.sqrt(B * B - 4 * A * C)) / (2 * A)

    print("Root1 = "+str(root1)+" Root2 = "+str(root2))
    #return root1,root2
    x_axis = []
    fx = []
    for i in range(0,10):
        x_axis.append(i)
        fx.append(np.float(root1))

    #find the classification accuracy
    # (x-root1)(a-root2)> 0 then class1 else class2
    w1w1= 0
    w2w2 = 0
    for i in range(0,count-1):
        f = ( x11[i] - root1 )*(x11[i]-root2)
        if f>0:
            w1w1 = w1w1+1
        else:
            w2w2 = w2w2+1

        f = ( x21[i] - root1 )*(x21[i]-root2)
        if f>0:
            w1w1 = w1w1+1
        else:
            w2w2 = w2w2+1

    accuracy = (w1w1+w2w2)/(count*2)*100

    return x_axis, fx,accuracy

#Plotting function, takes points and dimensions to plot
def plot_fishers(x11,x12,x21,x22, dim1,dim2,curve_x,curve_fx):
    # Getting points in proportions for plotting
    #finding minimum and maximum from each array
    minX11 = min(x11)
    maxX11 = max(x11)
    minX12 = min(x12)
    maxX12 = max(x12)
    propX11 = maxX11 - minX11
    propX12 = maxX12 - minX12

    minX21 = min(x21)
    maxX21 = max(x21)
    minX22 = min(x22)
    maxX22 = max(x22)
    propX21 = maxX21 - minX21
    propX22 = maxX22 - minX22
    prop_curve_x = ( max(curve_x) - min(curve_x))
    #prop_curve_fx = ( max(curve_fx) - min(curve_fx))
    #divide the array from the difference of minimum and maximum
    x11 = [x / propX11 for x in x11]
    x12 = [x / propX12 for x in x12]
    x21 = [x / propX21 for x in x21]
    x22 = [x / propX22 for x in x22]
    curve_x = [ x*2 / prop_curve_x for x in curve_x]
    curve_fx = [ x / (x+1) for x in curve_fx]
    #print(curve_fx)

    plt.scatter( x11 , x12, marker = "o" )
    plt.scatter( x21 , x22, marker = "*" )
    # Plot the curves
    plt.plot(curve_x, curve_fx, color = 'black', label = 'x2 first root' )

    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.hlines(0, -1.5, 1.5)
    plt.xlabel(dim1)
    plt.vlines(0, -1.5, 1.5)
    plt.ylabel(dim2)
    plt.show()


#this function calculates Euclidean distance
def euc_dist(x11,x12,x13,x14,x15,x16,x21,x22,x23,x24,x25,x26):
        d = pow((x11 - x21), 2 ) + pow((x12 - x22), 2 ) +pow((x13 - x23), 2 )+pow((x14 - x24), 2 )+pow((x15 - x25), 2) +pow((x16 - x26), 2 )
        return math.sqrt(d)


#this function is for K nearest neighbour where  k = 1
def kNN(x11,x12,x13,x14,x15,x16):

    euc_dist_list = []
    predicted = []

    for i in range(0,count-2):
        for j in range(i+1, count):
            a = euc_dist( x11[i],x12[i],x13[i],x14[i],x15[i],x16[i], x11[i+1],x12[i+1],x13[i+1],x14[i+1],x15[i+1],x16[i+1] )
            euc_dist_list.append(a)

        #print("Euc dist")
        #print(euc_dist_list)
        euc_dist_list.sort()
        predicted.append(euc_dist_list[0])
    return predicted


### Ho-Kashyap Functions
def findE(x, b):  # Function to find the matrix "E"
    x_inv = numpy.linalg.pinv(x)  # Find the Moore-Penrose Inverse of x
    W = numpy.dot(x_inv, b)  # Multiple x# and b
    E = numpy.subtract(numpy.dot(x, W), b)  # Subtract b from x times W
    E_rounded = numpy.around(E.astype(numpy.double), 1)  # Round the E matrix to one decimal point
    return E_rounded, W

# Is Matrix E equal to 0?
def is_E_zero(E):
    for i in E:
        if (i != 0):
            return False
    return True
#is Matrix E <0
def is_E_less_than_zero(E):
    for i in E:
        if (i > 0 or i == 0):
            return False
    return True

#this function will print the linear classifier
def print_equation(x11,x12,x21,x22,x13,x23,W):
    x = ((W[2] * -1) / W[0])
    print("\nClassification is possible \n Equation of the line is x =",x )
    # Print statement for 2D problems
    # print "Equation of the line is x - y + z =", ((W[3]*-1)/W[0]) # Print statement for 3D problem
    curve_x = []
    curve_fx = []
    for i in range(0,10):
        curve_x.append(i)
        curve_fx.append(x)
    #plot_Ho_Kashyap(x11, x12, x21, x22, 'X1', 'X2', curve_x, curve_fx)
    #plot_Ho_Kashyap(x11, x13, x21, x23, 'X1', 'X3', curve_x, curve_fx)


def plot_Ho_Kashyap(x11,x12,x21,x22, dim1, dim2,curve_x,curve_fx):
    # Getting points in proportions for plotting
    #finding minimum and maximum from each array
    minX11 = min(x11)
    maxX11 = max(x11)
    minX12 = min(x12)
    maxX12 = max(x12)
    propX11 = maxX11 - minX11
    propX12 = maxX12 - minX12

    minX21 = min(x21)
    maxX21 = max(x21)
    minX22 = min(x22)
    maxX22 = max(x22)
    propX21 = maxX21 - minX21
    propX22 = maxX22 - minX22
    prop_curve_x = ( max(curve_x) - min(curve_x))
    #prop_curve_fx = ( max(curve_fx) - min(curve_fx))
    #divide the array from the difference of minimum and maximum
    x11 = [x / propX11 for x in x11]
    x12 = [x / propX12 for x in x12]
    x21 = [x / propX21 for x in x21]
    x22 = [x / propX22 for x in x22]
    curve_x = [ x*2 / prop_curve_x for x in curve_x]
    curve_fx = [ x / (x+1) for x in curve_fx]
    #print(curve_fx)

    plt.scatter( x11 , x12, marker = "o" )
    plt.scatter( x21 , x22, marker = "*" )
    # Plot the curves
    plt.plot(curve_x, curve_fx, color = 'black', label = 'x2 first root' )

    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.hlines(0, -1.5, 1.5)
    plt.xlabel(dim1)
    plt.vlines(0, -1.5, 1.5)
    plt.ylabel(dim2)
    plt.show()


def main():

    ### Data Pre-processing..... ####
    # Fetching bank data from csv files
    colnames = ['Balance', 'Day', 'Duration', 'Campaign', 'Pdays', 'Age']
    data_married = pandas.read_csv(r'C:\Users\Adarsh\uOttawa\Courses\Winter18\SSPR\Project\Married.csv', names=colnames, sep=";")
    data_single = pandas.read_csv(r'C:\Users\Adarsh\uOttawa\Courses\Winter18\SSPR\Project\Single.csv', names=colnames, sep=";")

    balance_married = data_married.Balance.tolist()
    x11 = balance_married[1:count]
    x11 = list(map(float, x11))
    duration_married = data_married.Duration.tolist()
    x12 = duration_married[1:count]
    x12 = list(map(float, x12))
    age_married = data_married.Age.tolist()
    x13 = age_married[1:count]
    x13 = list(map(float, x13))
    campaign_married = data_married.Campaign.tolist()
    x14 = campaign_married[1:count]
    x14 = list(map(float, x14))
    pdays_married = data_married.Pdays.tolist()
    x15 = pdays_married[1:count]
    x15 = list(map(float, x15))
    day_married = data_married.Day.tolist()
    x16 = day_married[1:count]
    x16 = list(map(float, x16))

    balance_single = data_single.Balance.tolist()
    x21 = balance_single[1:count]
    x21 = list(map(float, x21))
    duration_single = data_single.Duration.tolist()
    x22 = duration_single[1:count]
    x22 = list(map(float, x22))
    age_single = data_single.Age.tolist()
    x23 = age_single[1:count]
    x23 = list(map(float, x23))
    campaign_single = data_single.Campaign.tolist()
    x24 = campaign_single[1:count]
    x24 = list(map(float, x24))
    pdays_single = data_single.Pdays.tolist()
    x25 = pdays_single[1:count]
    x25 = list(map(float, x25))
    day_single = data_single.Day.tolist()
    x26 = duration_single[1:count]
    x26 = list(map(float, x26))

    ## print out the points initially
    #plot(x11, x12, x21,x22, 'Balance ' , 'Duration')
    #plot(x11, x13, x21,x23, 'Balance' , 'Age ')

    ## find the Mean for both the classes Married and Single
    M_class1 = M_Married = np.matrix([[sum(x11)/count], [sum(x12)/count], [sum(x13)/count], [sum(x14)/count], [sum(x15)/count], [sum(x16)/count] ])
    M_class2 = M_Single = np.matrix(
        [[sum(x21) / count], [sum(x22) / count], [sum(x23) / count], [sum(x24) / count], [sum(x25) / count],
         [sum(x26) / count]])

    #print("Mean Married CLass1")
    #print(M_Married)
    #print("Mean Single Class2")
    #print(M_Single)

    X1 = [x11,x12,x13,x14,x15,x16]
    X2 = [x21,x22,x23,x24,x25,x26]

    Cov_Single = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]
                             ])
    Cov_Married =  np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]
                           ])


    Cov_Married = np.float64(Cov_Married)
    Cov_Single = np.float64(Cov_Single)
    #Cov_Single.astype(float)

    for i in range(0,6):
        for j in range(0,6):
            Cov_Married[i][j] = find_cov(X1[i], X1[j], M_Married[i], M_Married[j])
            Cov_Single[i][j] = find_cov(X2[i], X2[j], M_Single[i], M_Single[j])


    cov_class1 = Cov_Married
    cov_class2 = Cov_Single
    #print("Covariance of Class 1 (Married)")
    #print(Cov_Married)
    #print("Covariance of Class 2 (Single)")
    #print(Cov_Single)

    #### Maximum Likelihood
    Mean_class1_ML_est = np.matrix([[ sum(x11)], [sum(x12)],[sum(x13)],[ sum(x14)],[ sum(x15)],[ sum(x16)] ])
    Mean_class2_ML_est = np.matrix([[ sum(x21)], [sum(x22)],[sum(x23)],[ sum(x24)],[ sum(x25)],[ sum(x26)] ])

    #Mean1_ML_est = [x / count for x in Mean1_ML_est]
    #Mean2_ML_est = [x / count for x in Mean2_ML_est]
    Mean_class1_ML_est = divide_2d_matrix_by_n(Mean_class1_ML_est)
    Mean_class2_ML_est = divide_2d_matrix_by_n(Mean_class2_ML_est)
    #print("Estimated mean1 for Maximum Likeliood")
    #print(Mean_class1_ML_est)
    #print("Estimated mean2 for Maximum Likeliood")
    #print(Mean_class2_ML_est)

    summ = np.matrix([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]
                           ])
    for i in range(0,count-1):
        Xi = np.matrix([[ (x11[i])],[ (x12[i]) ],[(x13[i])], [(x14[i])],[(x15[i])],[(x16[i])] ])
        mult = ( (Xi - Mean_class1_ML_est) * np.matrix.transpose((Xi - Mean_class1_ML_est))  )
        summ = summ + mult

    #Cov1_ML_est = [x / count for x in summ]
    Cov_class1_ML_est = divide_2d_matrix_by_n(summ)
    #print("Cov1 Maximum Likeliood Estimated : ")
    #print(Cov_class1_ML_est)

    summ = np.matrix([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]
                           ])
    for i in range(0, count-1):
        Xi = np.matrix([[ (x21[i])],[ (x22[i]) ],[ (x23[i] )], [ (x24[i] )],[ (x25[i] )],[ (x26[i] )]  ])
        mult = ( (Xi - Mean_class2_ML_est) * np.matrix.transpose((Xi - Mean_class2_ML_est))  )
        summ = summ + mult

    #Cov1_ML_est = [x / count for x in summ]
    Cov_class2_ML_est = divide_2d_matrix_by_n(summ)
    #print("Cov2 Maximum Likeliood Estimated  : ")
    #print(Cov_class2_ML_est)


    ##### Baysian Metholdology - Calculating mean
    Sigma0 = np.matrix([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        ])

    cov_class1_BM = divide_2d_matrix_by_n(cov_class1)
    cov_class1_BM = cov_class1_BM + Sigma0
    cov_class1_BM = np.linalg.inv(cov_class1_BM)
    temp1 = cov_class1 * cov_class1_BM
    temp1 = temp1 * M_class1
    #temp1 = [ x / count for x in temp1 ]
    temp1 = divide_2d_matrix_by_n(temp1)

    temp2 = Sigma0 * cov_class1_BM
    summ = np.matrix([[0],
                     [0],
                      [0],
                      [0],
                      [0],
                     [0]])

    for i in range (0,count-1):
        Xi = np.matrix([[x11[i]],
                     [ x12[i] ],
                     [x13[i] ],
                        [x14[i]],
                        [x15[i]],
                        [x16[i]]
                        ])
        summ = summ + Xi

    summ = divide_2d_matrix_by_n(summ)
    #print("printing summ matrix")
    #print(summ)

    temp2 = temp2 * summ
    Mean_class1_BM_est = temp1 + temp2
    #print("Mean for Bayesian method. class1")
    #print(Mean_class1_BM_est)

    #class2
    cov_class2_BM = divide_2d_matrix_by_n(cov_class2)
    cov_class2_BM = cov_class2_BM + Sigma0
    cov_class2_BM = np.linalg.inv(cov_class2_BM)
    temp1 = cov_class2 * cov_class2_BM
    temp1 = temp1 * M_class2
    # temp1 = [ x / count for x in temp1 ]
    temp1 = divide_2d_matrix_by_n(temp1)

    temp2 = Sigma0 * cov_class2_BM
    summ = np.matrix([[0],
                     [0],
                      [0],
                      [0],
                      [0],
                     [0]])

    for i in range(0, count-1):
        Xi = np.matrix([[x21[i]],
                        [x22[i]],
                        [x23[i]],
                        [x24[i]],
                        [x25[i]],
                        [x26[i]]
                        ])
        summ = summ + Xi

    summ = divide_2d_matrix_by_n(summ)
    # print("printing summ matrix")
    # print(summ)

    temp2 = temp2 * summ
    Mean_class2_BM_est = temp1 + temp2
    #print("Mean for Bayesian method class2")
    #print(Mean_class2_BM_est)
    Cov_class1_BM_est = Cov_Married
    Cov_class2_BM_est = Cov_Single
    #Cov1_BM_est
    #print("Cov Matrix for Bayesian method class1")
    #print(Cov_class1_BM_est)
    #print("Cov Matrix for Bayesian method class2")
    #print(Cov_class2_BM_est)


    #### Parzen Window
    Mean_class1_Par = []
    Mean_class2_Par = []
    Var1_Par = []
    Var2_Par = []

    for i in range(0,6):
        [a,b] = uni_var_parzen(X1[i], M_class1[0], cov_class1.item(i, i))
        Mean_class1_Par.append(a)
        Var1_Par.append(b)
        #print("for class1, X1"+str(i+1)+", Sample Mean = " + str(a) + " and variance  = " + str(b))
        [a,b] = uni_var_parzen(X2[i], M_class2[0], cov_class2.item(i, i))
        Mean_class2_Par.append(a)
        Var2_Par.append(b)
        #print("for class2, X2"+str(i+1)+", Sample Mean = " + str(a) + " and variance  = " + str(b))


    ## Bayes Descriminant function for Maximum Likelihood
    #calc_bayes(Mean_class1_ML_est, Mean_class2_ML_est, Cov_class1_ML_est, Cov_class2_ML_est, x11, x12, x13, x21, x22, x23)
    ## Bayes Descriminant function for BM
    #calc_bayes(Mean_class1_BM_est, Mean_class2_BM_est, Cov_class1_BM_est, Cov_class2_BM_est, x11, x12, x13, x21, x22, x23)

    ## Test points
    x11_test = balance_married[count:count*2]
    x11_test = list(map(int, x11_test))
    x12_test = duration_married[count:count*2]
    x12_test = list(map(int, x12_test))
    x13_test = age_married[count:count*2]
    x13_test = list(map(int, x13_test))

    x21_test = balance_single[count:count*2]
    x21_test = list(map(int, x21_test))
    x22_test = duration_single[count:count*2]
    x22_test = list(map(int, x22_test))
    x23_test = age_single[count:count*2]
    x23_test = list(map(int, x23_test))

    #class_accuracy( Mean_class1_ML_est, Mean_class2_ML_est, Cov_class1_ML_est, Cov_class2_ML_est,x11,x12,x13,x14,x15,x16,x21,x22,x23,x24,x25,x26 )
    #class_accuracy(Mean_class1_BM_est, Mean_class2_BM_est, Cov_class1_BM_est, Cov_class2_BM_est, x11,x12,x13,x14,x15,x16,x21,x22,x23,x24,x25,x26)
    #class_accuracy(Mean_class1_Par, Mean_class2_Par, Cov_class1_Par, Cov_class2_Par, x11_test, x12_test, x13_test, x21_test, x22_test, x23_test)


    ##### kNN ipmlementation
    class1 = kNN(x11,x12,x13,x14,x15,x16)
    class2 = kNN(x21,x22,x23,x24,x25,x26)
    print("After CLassification with 1-Nearest Neighbour...")
    print("Correctly classified in Class1 = "+ str(len(class1)))
    print("Correctly classified in Class2 = " + str(len(class2)))
    accuracy = ( len(class1) + len(class2) )/ (count*2) *100
    print("Classification Accuracy with 1-Nearest NEigbour = "+str(accuracy))


    #### Fisher's Descriminant
    [x_axis, fx,accuracy] = fisher_desc(Mean_class1_BM_est,Mean_class2_BM_est, Cov_class1_BM_est, Cov_class2_BM_est,x11,x21)
    #plot_fishers(x11, x12, x21, x22, 'X1', 'X2', x_axis, fx)
    #plot_fishers(x11, x13, x21, x23, 'X1', 'X3', x_axis, fx)
    #print("Accuracy of Fisher's Descriminant : ")
    #print(accuracy)


    ### Ho-Kashyap's ALgorithm
    X  = np.concatenate((X1,X2), axis=1)
    b = [1, 1, 1, 1, 1, 1]
    while (True):  # Keep running until we find a solution or a solution is not possible
        E, W = findE(X, b)
        E_zero= is_E_zero(E)
        E_lessthan_zero = is_E_less_than_zero(E)

        if (E_zero):
            print_equation(x11,x12,x21,x22,x13,x23,W)
            break
        elif (E_lessthan_zero):
            print("There is no solution possible")
            break
        b = numpy.add(b, numpy.add(E, numpy.absolute(E)))  # Add b to the addition of E and the absolute value of E



if __name__ == "__main__":
        main()

