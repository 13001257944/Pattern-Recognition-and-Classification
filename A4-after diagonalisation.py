import math
import numpy as np
import matplotlib.pyplot as plt

# defining globally the number of count of the total numbers need to be generated
count = 200

def set_cov():
    [cov1, cov2] = get_covariance_matrix(5, 3, 4, 0.1, 0.2)
    return cov1,cov2

#this function will generate a single Gaussian random number
def single_gaussian_random():
    n = 0
    while n == 0:
        n = round(np.random.random() * 100)

    numbers = np.random.random(int(n))
    summation = float(np.sum(numbers))
    #Use central limit theory to generate one quassian number
    gaussian = (summation - n/2) / math.sqrt(n/12.0)
    return gaussian

#generate several n-dimensional Gaussian random numbers with a zero mean and identity covariance
def generate_gaussian(dimensions,count):

    #Get an empty list which will store the points
    list = []
    for i in range(0, count):
        current_vector = []
        for j in range(0, dimensions):
            g = single_gaussian_random()
            current_vector.append(g)

        list.append( tuple(current_vector) )

    return list

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

    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.hlines(0, -1.5, 1.5)
    plt.xlabel(dim1)
    plt.vlines(0, -1.5, 1.5)
    plt.ylabel(dim2)
    plt.show()

#Plot the convergence of the parameters
def plot_conv(X,Y,dimX,dimY):

    # Plot the curves
    plt.plot(X, Y , color='black', label='')
    plt.plot(X, Y, color='green', label='')

    #plt.axis([-1.5, 1.5, -1.5, 1.5])
    #plt.hlines(0, -1.5, 1.5)
    plt.xlabel(dimX)
    #plt.vlines(0, -1.5, 1.5)
    plt.ylabel(dimY)
    plt.show()


# get covariance matrix by feeding values of a,b,c,alpha,beta
def get_covariance_matrix(a,b,c,alpha,beta):

    Sigma1 = np.matrix([[a*a, beta*a*b, alpha*a*c],
                        [beta*a*b, b*b, beta*b*c],
                        [alpha*a*c, beta*b*c, c*c]])

    Sigma2 = np.matrix([[c*c, alpha*b*c, beta*a*c],
                        [alpha*b*c, b*b, alpha*a*b],
                        [beta*a*c, alpha*a*b, a*a]])

    return Sigma1,Sigma2

#Generate points
def generate_points(cov, mean):
    known1 = generate_gaussian(3, count)

    [eigenvalues1, eigenvectors1] = np.linalg.eig(cov)
    Lsqrt1 = np.matrix(np.diag(np.sqrt(eigenvalues1)))
    P1 = Lsqrt1 * np.matrix(eigenvectors1)

    x1_tweaked = []
    x2_tweaked = []
    x3_tweaked = []
    tweaked_all = []
    for i, j, k in known1:
        original1 = np.matrix([[i], [j], [k]]).copy()

        tweaked1 = (P1 * original1) + mean

        x1_tweaked.append(float(tweaked1[0]))
        x2_tweaked.append(float(tweaked1[1]))
        x3_tweaked.append(float(tweaked1[2]))
        tweaked_all.append(tweaked1)

    return x1_tweaked, x2_tweaked, x3_tweaked

def divide_2d_matrix_by_n(m):
    #print("printing matrix in divide function")

    for i in range(0,3,1):
        #print("printing "+str(m[i]))
        m[i] = [x / count for x in m[i]]
        #print("after divide "+str(m[i]))

    return m

def uni_var_parzen(X, mean, sigma):
    fx_values = []
    N = []
    summ = 0
    for i in range(0,200):
        exp_term = (X[i] - mean ) * ( X[i] -mean)
        exp_term = exp_term / ( 2*sigma*sigma )
        exp_term = np.exp((-exp_term))
        fx = (1/ (sigma*np.sqrt((2*3.14)) )) * (exp_term)
        fx = np.float64(fx)
        fx_values.append(fx)
        N.append(i)
        summ = summ + fx

    #Sample mean and variance
    sample_mean = (summ)/count
    sum = 0
    for i in range(0,200):
        #for variance calc
        diff = (X[i] - sample_mean)*(X[i] - sample_mean)
        sum = sum + diff

    sample_var = sum / (count-1)
    # Plot the curves
    plt.plot(N, fx_values , color='black', label='')
    plt.xlabel("N")
    plt.ylabel("Fx values")
    plt.show()
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

    prop_curve_x1 = ( max(curve_x22) - min(curve_x22))
    prop_curve_x21 = ( max(curve_x22) - min(curve_x22))
    prop_curve_x22 = ( max(curve_x22) - min(curve_x22))

    #divide the array from the difference of minimum and maximum
    x11 = [x / propX11 for x in x11]
    x12 = [x / propX12 for x in x12]
    x21 = [x / propX21 for x in x21]
    x22 = [x / propX22 for x in x22]
    curve_x1 = [ x*2 / prop_curve_x1 for x in curve_x1]
    curve_x21 = [ x*2 / prop_curve_x21 for x in curve_x21]
    curve_x22 = [ x*2 / prop_curve_x22 for x in curve_x22]

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

def calc_bayes(M1, M2, cov1, cov2, x11, x12, x13, x21, x22, x23 ):
    # Ax, Bx, Cx are the values of A,B,C in X domain
    Ax = (np.linalg.inv(cov2) - np.linalg.inv(cov1)) / 2
    Bx = ((np.matrix.transpose(M1) * np.linalg.inv(cov1)) - (np.matrix.transpose(M2) * np.linalg.inv(cov2)))
    # As apriori probabilities P1 and P2 are equal because of distribution, so term log P1/P2 would be zero
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


def main():
    ### Diagonalization
    global cov_class1, cov_class2
    # [cov_class1, cov_class2] = set_cov()

    M_class1_old = np.matrix([[8.0], [7.0], [11.0]])
    M_class2_old = np.matrix([[-13], [-2], [-4]])

    cov_old1 = np.matrix([[25.0, 3.0, 2.0],
                          [3.0, 9.0, 2.4],
                          [2.0, 2.4, 1.6]])
    cov_old2 = np.matrix([[16.0, 1.2, 4.0],
                          [1.2, 9.0, 1.5],
                          [4.0, 1.5, 25.0]])

    ### Simulatneous diagonalization, Generating Mean and Covariance matrix for Z
    [eigenvalues1, eigenvectors1] = np.linalg.eig(cov_old1)
    MeanZ1 = np.matrix(np.diag(1 / np.sqrt(eigenvalues1))) * np.matrix.transpose(eigenvectors1) * M_class1_old
    SigmaZ1 = I = np.matrix([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])

    [eigenvaluesZ1, eigenvectorsZ1] = np.linalg.eig(SigmaZ1)

    # Calculate for Z2
    MeanZ2 = np.matrix(np.diag(1 / np.sqrt(eigenvalues1))) * np.matrix.transpose(eigenvectors1) * M_class2_old
    SigmaZ2 = np.matrix(np.diag(1 / np.sqrt(eigenvalues1))) * np.matrix.transpose(eigenvectors1) * cov_old2 * (
            eigenvectors1 * np.matrix(np.diag(1 / np.sqrt(eigenvalues1))))

    [eigenvaluesZ2, eigenvectorsZ2] = np.linalg.eig(SigmaZ2)
    # End calculations for Z

    # Final transformation, Generating Mean and Variances for V
    MeanV1 = np.matrix.transpose(eigenvectorsZ2) * MeanZ1
    SigmaV1 = np.matrix.transpose(eigenvectorsZ2) * I * eigenvectorsZ2
    [eigenvaluesV1, eigenvectorsV1] = np.linalg.eig(SigmaV1)
    LsqrtV1 = np.matrix(np.diag(np.sqrt(eigenvaluesV1)))
    PV1 = np.matrix(eigenvectorsV1) * LsqrtV1

    MeanV2 = np.matrix.transpose(eigenvectorsZ2) * MeanZ2
    SigmaV2 = np.matrix.transpose(eigenvectorsZ2) * SigmaZ2 * eigenvectorsZ2
    [eigenvaluesV2, eigenvectorsV2] = np.linalg.eig(SigmaV2)
    LsqrtV2 = np.matrix(np.diag(np.sqrt(eigenvaluesV2)))
    PV2 = np.matrix(eigenvectorsV2) * LsqrtV2

    x11, x12, x13 = generate_points(SigmaV1, MeanV1)
    x21, x22, x23 = generate_points(SigmaV2, MeanV2)
    M_class1 = MeanV1
    M_class2 = MeanV2
    cov_class1 = SigmaV1
    cov_class2 = SigmaV2


    #### Answer A ####
    #Generate and plot 200 points

    #########  Generating the points
    #x11, x12, x13 = generate_points(cov_class1, M_class1)
    #x21, x22, x23 = generate_points(cov_class2, M_class2)

    #Plot points and Classification Curve for X in x1-x2 and x1-x3 dimensions
    plot(x11,x12,x21,x22,'X1', 'X2')
    plot(x11,x13,x21,x23,'X1', 'X3')

    #####  Answer b  ######
    # Maximum Likelihood
    p1 = int(sum(x11))
    p2 =  sum(x12)
    p3 = sum(x13)
    Mean_class1_ML_est = np.matrix( [ [p1],
                                [p2],
                                [p3] ]
                              )
    Mean_class2_ML_est = np.matrix([[ sum(x21)], [sum(x22)],[sum(x23)]])

    #Mean1_ML_est = [x / count for x in Mean1_ML_est]
    #Mean2_ML_est = [x / count for x in Mean2_ML_est]
    Mean_class1_ML_est = divide_2d_matrix_by_n(Mean_class1_ML_est)
    Mean_class2_ML_est = divide_2d_matrix_by_n(Mean_class2_ML_est)
    print("Estimated mean1 for ML")
    print(Mean_class1_ML_est)
    print("Estimated mean2 for ML")
    print(Mean_class2_ML_est)

    summ = np.matrix([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])

    for i in range(0,200):
        Xi = np.matrix([[ (x11[i])],[ (x12[i]) ],[ (x13[i] )]])
        mult = ( (Xi - Mean_class1_ML_est) * np.matrix.transpose((Xi - Mean_class1_ML_est))  )
        summ = summ + mult

    #Cov1_ML_est = [x / count for x in summ]
    Cov_class1_ML_est = divide_2d_matrix_by_n(summ)
    print("Cov1 ML est")
    print(Cov_class1_ML_est)

    summ = np.matrix([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])

    for i in range(0,200):
        Xi = np.matrix([[ (x21[i])],[ (x22[i]) ],[ (x23[i] )]])
        mult = ( (Xi - Mean_class2_ML_est) * np.matrix.transpose((Xi - Mean_class2_ML_est))  )
        summ = summ + mult

    #Cov1_ML_est = [x / count for x in summ]
    Cov_class2_ML_est = divide_2d_matrix_by_n(summ)
    print("Cov2 ML est")
    print(Cov_class2_ML_est)


    # Baysian Metholdology - Calculating mean
    Sigma0 = np.matrix([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

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
                     [0]])

    for i in range (0,200):
        Xi = np.matrix([[x11[i]],
                     [ x12[i] ],
                     [x13[i] ]])
        summ = summ + Xi

    summ = divide_2d_matrix_by_n(summ)
    #print("printing summ matrix")
    #print(summ)

    temp2 = temp2 * summ
    Mean_class1_BM_est = temp1 + temp2
    print("Mean for BM class1")
    print(Mean_class1_BM_est)

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
                      [0]])

    for i in range(0, 200):
        Xi = np.matrix([[x21[i]],
                        [x22[i]],
                        [x23[i]]])
        summ = summ + Xi

    summ = divide_2d_matrix_by_n(summ)
    # print("printing summ matrix")
    # print(summ)

    temp2 = temp2 * summ
    Mean_class2_BM_est = temp1 + temp2
    print("Mean for BM class2")
    print(Mean_class2_BM_est)

    #Cov1_BM_est
    Cov_class1_BM_est = np.matrix([[20, 4, 1],
                        [2, 15, -1],
                        [-1, -1, 8]])

    print("Cov Matrix for BM class1")
    print(Cov_class1_BM_est)

    Cov_class2_BM_est = np.matrix([[10, 2, -1],
                        [2, 8, -2],
                        [1, 1, 5]])

    print("Cov Matrix for BM class2")
    print(Cov_class2_BM_est)

    #Plot the convergence of parameters with the number of smaples
    # Convergence of Mean for ML for class1 and class2
    Mean_diff_ML_class1 = abs( Mean_class1_ML_est - M_class1 )
    print(Mean_diff_ML_class1)
    Mean_diff_ML_class2 = abs( Mean_class2_ML_est - M_class2 )
    print(Mean_diff_ML_class2)
    N = np.matrix([[1],
                      [2],
                      [3]])

    #plot
    plot_conv(N, Mean_diff_ML_class1, 'N', 'ML Means class1')
    plot_conv(N, Mean_diff_ML_class2, 'N', 'ML Means class2')

    # Convergence of Covariances for ML for class1 and class2
    Cov_diff_ML_class1 = abs( cov_class1 - Cov_class1_ML_est )
    Cov_diff_ML_class2 = abs( cov_class2 - Cov_class2_ML_est )
    plot_conv(N, Cov_diff_ML_class1, 'N', 'ML Cov class1')
    plot_conv(N, Cov_diff_ML_class2, 'N', 'ML Cov class2')

    # Convergence of Mean for BM for class1 and class2
    Mean_diff_BM_class1 = abs( Mean_class1_BM_est - M_class1 )
    print(Mean_diff_BM_class1)
    Mean_diff_BM_class2 = abs( Mean_class2_BM_est - M_class2 )
    print(Mean_diff_BM_class2)
    N = np.matrix([[1],
                      [2],
                      [3]])

    plot_conv(N, Mean_diff_BM_class1, 'N', 'BM Means class1')
    plot_conv(N, Mean_diff_BM_class2, 'N', 'BM Means class2')

    # Convergence of Covariances for ML for class1 and class2
    Cov_diff_BM_class1 = abs( cov_class1 - cov_class1_BM )
    Cov_diff_BM_class2 = abs( cov_class2 - cov_class2_BM )
    plot_conv(N, Cov_diff_BM_class1, 'N', 'BM Cov class1')
    plot_conv(N, Cov_diff_BM_class2, 'N', 'BM Cov class2')




    ### Answer C ###

    [m_class1_row1, cov_class1_row1] = uni_var_parzen(x11, M_class1[0], cov_class1.item(0,0))
    print( "for class1, X1, Sample Mean = "+str(m_class1_row1)+" and variance  = "+str(cov_class1_row1))
    [m_class1_row2, cov_class1_row2] = uni_var_parzen(x12, M_class1[1], cov_class1.item(1,1))
    print("for class1 ,X2, Sample Mean = "+str(m_class1_row2)+" and variance  = "+str(cov_class1_row2))
    [m_class1_row3, cov_class1_row3] = uni_var_parzen(x13, M_class1[2], cov_class1.item(2,2))
    print("for class3, X3, Sample Mean = "+str(m_class1_row3)+" and variance  = "+str(cov_class1_row3))

    [m_class2_row1, cov_class2_row1] = uni_var_parzen(x21, M_class2[0], cov_class2.item(0,0))
    print("for class 2, X1, Sample Mean = "+str(m_class2_row1)+" and variance  = "+str(cov_class2_row1))
    [m_class2_row2, cov_class2_row2] = uni_var_parzen(x22, M_class2[1], cov_class2.item(1,1))
    print("for class 2, X2, Sample Mean = "+str(m_class2_row2)+" and variance  = "+str(cov_class2_row2))
    [m_class2_row3, cov_class2_row3] = uni_var_parzen(x23, M_class2[2], cov_class2.item(2,2))
    print("for class 2 , X3, Sample Mean = "+str(m_class2_row3)+" and variance  = "+str(cov_class2_row3))

    Mean_class1_Par = np.matrix([[m_class1_row1],
                                [m_class1_row2],
                                [m_class1_row3]])

    Mean_class2_Par = np.matrix([[m_class2_row1],
                                [m_class2_row2],
                                [m_class2_row3]])


    Cov_class1_Par = cov_class1
    Cov_class2_Par = cov_class2
    #Cov_class1_Par[0][0] = cov_class1_row1
    #Cov_class1_Par[1][1] = cov_class1_row2
    #Cov_class1_Par[2][2] = cov_class1_row3

    #Cov_class2_Par[0][0] = cov_class2_row1
    #Cov_class2_Par[1][1] = cov_class2_row2
    #Cov_class2_Par[2][2] = cov_class2_row3




    ### Answer d ###
    ## Bayes Descriminant function for ML
    calc_bayes(Mean_class1_ML_est, Mean_class2_ML_est, Cov_class1_ML_est, Cov_class2_ML_est, x11, x12, x13, x21, x22, x23)
    ## Bayes Descriminant function for BM
    calc_bayes(Mean_class1_BM_est, Mean_class2_BM_est, Cov_class1_BM_est, Cov_class2_BM_est, x11, x12, x13, x21, x22, x23)
    ## Bayes Descriminant function for Parzen Window
    calc_bayes(Mean_class1_Par, Mean_class2_Par, Cov_class1_Par, Cov_class2_Par, x11, x12, x13, x21, x22, x23)




if __name__ == "__main__":
        main()

