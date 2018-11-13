import math
import numpy as np
import matplotlib.pyplot as plt

#defining Mean1 and Mean2 globally
M1 = np.matrix([[8.0], [7.0], [11.0]])
M2 = np.matrix([[-13], [-2], [-4]])

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
def plot(x11,x12,x21,x22, dim1,dim2,curve_x1,curve_x21,curve_x22):
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


def main():

    #### Answer A ####
    #Generate and plot 200 points

    global cov1,cov2
    [cov1, cov2] = set_cov()
    print("Mean1 = ")
    print(M1)
    print("Mean2 = ")
    print(M2)
    print("Covariance1 = ")
    print(cov1)
    print("Covariance2 =")
    print(cov2)

    #########  Generating the points
    x11, x12, x13 = generate_points(cov1, M1)
    x21, x22, x23 = generate_points(cov2, M2)

    #### Answer B ####
    # Creating the optimal Bayes Discriminant function
    # Drwaing the curves for classification

    #Ax, Bx, Cx are the values of A,B,C in X domain
    Ax = ( np.linalg.inv(cov2) - np.linalg.inv(cov1)  )/2
    Bx = ( ( np.matrix.transpose(M1)*np.linalg.inv(cov1) ) - ( np.matrix.transpose(M2)*np.linalg.inv(cov2) ) )
    # As apriori probabilities P1 and P2 are equal because of distribution, so term log P1/P2 would be zero
    Cx = np.log( np.linalg.det(cov2) / np.linalg.det(cov1) )

    print("The equation of the quadratic classifier :")
    print("fx = ( np.matrix.transpose(X) * Ax * X ) + ( np.matrix.transpose(Bx)*X ) + Cx")
    print("Where Ax = ")
    print(Ax)
    print("Where Bx = ")
    print(Bx)
    print("Where Cx = ")
    print(Cx)

    #quadratic equation for X1 - X2, and solving it for x2 while feeding the values of x1
    # Getting the values of x2 by the formula ( -b +/- srt (b*b - 4*a*c) ) / (2*a)
    a = Ax.item(1,1)
    x2_root1 = []
    x2_root2 = []
    x12_values = []
    #feeding values of i for x1
    for i in range(-10, 15,1):
            b = (Ax.item(1, 0) * i) + (Ax.item(0, 1) * i) + Bx.item(1)
            c = (Ax.item(0, 0) * i * i) + (Bx.item(0) * i) + Cx
            if (( b * b - 4 * a * c ) > 0 ):
                x12_values.append(i)
                x2_root1.append((- b + np.sqrt(b * b - 4 * a * c)) / (2 * a))
                x2_root2.append((- b - np.sqrt(b * b - 4 * a * c)) / (2 * a))
            else:
                continue

    # quadratic equation for X1 - X3, and solving it for x3 while feeding the values of x1
    # Getting the values of x3 by the formula ( -b +/- srt (b*b - 4*a*c) ) / (2*a)
    a = Ax.item(2,2)
    x3_root1 = []
    x3_root2 = []
    x13_values = []
    #feeding values of i for x1
    for i in range(-10, 15,1):
            b = (Ax.item(2, 0) * i) + (Ax.item(0, 2) * i) + Bx.item(2)
            c = (Ax.item(0, 0) * i * i) + (Bx.item(0) * i) + Cx

            if ( (b * b - 4 * a * c) > 0):
                x13_values.append(i)
                x3_root1.append((- b + np.sqrt(b * b - 4 * a * c)) / (2 * a))
                x3_root2.append((- b - np.sqrt(b * b - 4 * a * c)) / (2 * a))
            else:
                continue

    #Plot points and Classification Curve for X in x1-x2 and x1-x3 dimensions
    plot(x11,x12,x21,x22,'X1', 'X2', x12_values, x2_root1,x2_root2)
    plot(x11,x13,x21,x23,'X1', 'X3', x13_values, x3_root1,x3_root2)


    #### Answer C ####
    ### Genrate 200 new points and Test classification accuracy

    x11_test, x12_test, x13_test = generate_points(cov1, M1)
    x21_test, x22_test, x23_test = generate_points(cov2, M2)

    # Test and Build a decesion matrix
    # Decision matrix first element true- true for class w1, and fourth element is true-ture for w2

    True_count_Xw1w1 = 0
    True_count_Xw2w2 = 0

    for i in range(0, count):
        Xx1 = np.matrix( [[x11_test[i]], [x12_test[i]] , [x13_test[i]]] )
        fn = ( (np.matrix.transpose(Xx1) * Ax)*Xx1 ) + ( (Bx) * Xx1 ) + Cx

        # check the value of expression for X, if it is Greater than 0, point X belongs to class w1,
        # else it belongs to w2
        if fn > 0:
            True_count_Xw1w1 = True_count_Xw1w1 +1
        else:
            True_count_Xw2w2 = True_count_Xw2w2 + 1

        Xx2 = np.matrix( [[x21_test[i]], [x22_test[i]] , [x23_test[i]]] )
        fn = ( (np.matrix.transpose(Xx2) * Ax) * Xx2 ) + (Bx * Xx2) + Cx

        if fn > 0:
            True_count_Xw1w1 = True_count_Xw1w1 +1
        else:
            True_count_Xw2w2 = True_count_Xw2w2 + 1

    #calculate accuracy
    accuracy = ( ( True_count_Xw1w1 + True_count_Xw2w2 ) / 400 ) * 100

    print("c. The accuracy Before diagonalization is : ")
    print(accuracy)


    #### Answer D ####
    ### Diagonalization

    ### Simulatneous diagonalization, Generating Mean and Covariance matrix for Z
    [eigenvalues1, eigenvectors1] = np.linalg.eig(cov1)
    MeanZ1 = np.matrix(np.diag(1/np.sqrt(eigenvalues1)))* np.matrix.transpose(eigenvectors1)*M1
    SigmaZ1 = I=  np.matrix([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])

    [eigenvaluesZ1, eigenvectorsZ1] = np.linalg.eig(SigmaZ1)


    #Calculate for Z2
    MeanZ2 = np.matrix(np.diag(1/np.sqrt(eigenvalues1)))* np.matrix.transpose(eigenvectors1)*M2
    SigmaZ2 = np.matrix(np.diag(1/np.sqrt(eigenvalues1)))* np.matrix.transpose(eigenvectors1)*cov2 * (eigenvectors1*np.matrix(np.diag(1/np.sqrt(eigenvalues1))))

    [eigenvaluesZ2, eigenvectorsZ2] = np.linalg.eig(SigmaZ2)

    print("Covariance matrix for Z1")
    print(SigmaZ1)
    print("Covariance matrix for Z2")
    print(SigmaZ2)
    #End calculations for Z

    # Final transformation, Generating Mean and Variances for V
    MeanV1 = np.matrix.transpose(eigenvectorsZ2)*MeanZ1
    SigmaV1 = np.matrix.transpose(eigenvectorsZ2)* I * eigenvectorsZ2
    [eigenvaluesV1, eigenvectorsV1] = np.linalg.eig(SigmaV1)
    LsqrtV1 = np.matrix(np.diag(np.sqrt(eigenvaluesV1)))
    PV1 = np.matrix(eigenvectorsV1) * LsqrtV1


    MeanV2 = np.matrix.transpose(eigenvectorsZ2)*MeanZ2
    SigmaV2 = np.matrix.transpose(eigenvectorsZ2)* SigmaZ2* eigenvectorsZ2
    [eigenvaluesV2, eigenvectorsV2] = np.linalg.eig(SigmaV2)
    LsqrtV2 = np.matrix(np.diag(np.sqrt(eigenvaluesV2)))
    PV2 = np.matrix(eigenvectorsV2) * LsqrtV2

    print("Cavariance matrix for V1:-")
    print(SigmaV1)
    print("Cavariance matrix for V2:-")
    print(SigmaV2)

    v11, v12, v13 = generate_points(SigmaV1, MeanV1)
    v21, v22, v23 = generate_points(SigmaV2, MeanV2)

    #### Answer E ####

    ##########  Drwaing the curves for classification

    # Ax, Bx, Cx are the values of A,B,C in X domain
    Av = (np.linalg.inv(SigmaV2) - np.linalg.inv(SigmaV1)) / 2
    Bv = ((np.matrix.transpose(MeanV1) * np.linalg.inv(SigmaV1)) - (np.matrix.transpose(MeanV2) * np.linalg.inv(SigmaV2)))
    # As apriori probabilities P1 and P2 are equal because of distribution, so term log P1/P2 would be zero
    Cv = np.log(np.linalg.det(SigmaV2) / np.linalg.det(SigmaV1))

    print("The equation of the quadratic classifier in V domain:")
    print("fv = ( np.matrix.transpose(V) * Av * V ) + ( np.matrix.transpose(Bv)*V ) + Cv")
    print("Where Av = ")
    print(Av)
    print("Where Bv = ")
    print(Bv)
    print("Where Cv = ")
    print(Cv)

    # the equation of quadratic classifier
    # fx = ( np.matrix.transpose(X) * A * X ) + ( np.matrix.transpose(B)*X ) + C

    # quadratic equation for X1 - X2, and solving it for x2 while feeding the values of x1
    # Getting the values of x2 by the formula ( -b +/- srt (b*b - 4*a*c) ) / (2*a)
    a = Av.item(1, 1)
    v2_root1 = []
    v2_root2 = []
    v12_values = []
    # feeding values of i for x1
    for i in range(-10, 15, 1):
        b = (Av.item(1, 0) * i) + (Av.item(0, 1) * i) + Bv.item(1)
        c = (Av.item(0, 0) * i * i) + (Bv.item(0) * i) + Cv
        if ((b * b - 4 * a * c) > 0):
            v12_values.append(i)
            v2_root1.append((- b + np.sqrt(b * b - 4 * a * c)) / (2 * a))
            v2_root2.append((- b - np.sqrt(b * b - 4 * a * c)) / (2 * a))
        else:
            continue

    # quadratic equation for X1 - X3, and solving it for x3 while feeding the values of x1
    # Getting the values of x3 by the formula ( -b +/- srt (b*b - 4*a*c) ) / (2*a)
    a = Av.item(2, 2)
    v3_root1 = []
    v3_root2 = []
    v13_values = []
    # feeding values of i for x1
    for i in range(-15, 15, 1):
        b = (Av.item(2, 0) * i) + (Av.item(0, 2) * i) + Bv.item(2)
        c = (Av.item(0, 0) * i * i) + (Bv.item(0) * i) + Cv

        if ((b * b - 4 * a * c) > 0):
            v13_values.append(i)
            v3_root1.append((- b + np.sqrt(b * b - 4 * a * c)) / (2 * a))
            v3_root2.append((- b - np.sqrt(b * b - 4 * a * c)) / (2 * a))
        else:
            continue

    # Plot points and Classification Curve for X in x1-x2 and x1-x3 dimensions
    plot(v11,v12,v21,v22,'V1', 'V2', v12_values, v2_root1,v2_root2)
    plot(v11,v13,v21,v23,'V1', 'V3', v13_values, v3_root1,v3_root2)


    ##### Answer F  ####

    # Test and Build a decesion matrix for V domain using the same points used in c
    # Decision matrix first element true- true for class w1, and fourth element is true-ture for w2

    True_count_Vw1w1 = 0
    True_count_Vw2w2 = 0

    for i in range(0, count):
        Xx1 = np.matrix( [[x11_test[i]], [x12_test[i]] , [x13_test[i]]] )
        fn = ( (np.matrix.transpose(Xx1) * Av)*Xx1 ) + ( (Bv) * Xx1 ) + Cv

        # check the value of expression for X, if it is Greater than 0, point X belongs to class w1,
        # else it belongs to w2
        if fn > 0:
            True_count_Vw1w1 = True_count_Vw1w1 +1
        else:
            True_count_Vw2w2 = True_count_Vw2w2 + 1

        Xx2 = np.matrix( [[x21_test[i]], [x22_test[i]] , [x23_test[i]]] )
        fn = ( (np.matrix.transpose(Xx2) * Av) * Xx2 ) + (Bv * Xx2) + Cv

        if fn > 0:
            True_count_Vw1w1 = True_count_Vw1w1 +1
        else:
            True_count_Vw2w2 = True_count_Vw2w2 + 1

    #calculate accuracy
    accuracy = ( ( True_count_Vw1w1 + True_count_Vw2w2 ) / 400 ) * 100

    print("e. The accuracy AFTER diagonalization is : ")
    print(accuracy)



if __name__ == "__main__":
    main()
