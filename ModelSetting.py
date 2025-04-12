from cmath import sqrt 
from operator import itemgetter 
from turtle import pos 
import gurobipy as gp 
from gurobipy import GRB 
import math 
pi = math.pi 
import numpy as np 
import pandas as pd 
import dill 
import time

# plotting libraries 
from mpl_toolkits import mplot3d 
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


last_save_time = time.time()
save_interval = 3600  


def my_callback(model, where):#save data from time to time
    global last_save_time
    if where == GRB.Callback.MIPSOL:  
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            last_save_time = current_time  
            
        
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            
        
            xMatrix0 = np.zeros((N, A))
            xMatrix1 = np.zeros((N, A))
            xMatrix2 = np.zeros((N, A)) if D > 2 else None
            
            vMatrix0 = np.zeros((N, A))
            vMatrix1 = np.zeros((N, A))
            vMatrix2 = np.zeros((N, A)) if D > 2 else None
            
        
            for i in range(N):
                for ac in range(A):
                    xMatrix0[i, ac] = model.cbGetSolution(x[0, i, ac])
                    xMatrix1[i, ac] = model.cbGetSolution(x[1, i, ac])
                    if D > 2:
                        xMatrix2[i, ac] = model.cbGetSolution(x[2, i, ac])
                    
                    vMatrix0[i, ac] = model.cbGetSolution(v[0, i, ac])
                    vMatrix1[i, ac] = model.cbGetSolution(v[1, i, ac])
                    if D > 2:
                        vMatrix2[i, ac] = model.cbGetSolution(v[2, i, ac])

        
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            dfx0 = pd.DataFrame(xMatrix0)
            dfx0.to_csv(f"{WsFileName}{timestamp}_xMatrix0.csv")
            dfx1 = pd.DataFrame(xMatrix1)
            dfx1.to_csv(f"{WsFileName}{timestamp}_xMatrix1.csv")
            if D > 2:
                dfx2 = pd.DataFrame(xMatrix2)
                dfx2.to_csv(f"{WsFileName}{timestamp}_xMatrix2.csv")

            dfv0 = pd.DataFrame(vMatrix0)
            dfv0.to_csv(f"{WsFileName}{timestamp}_vMatrix0.csv")
            dfv1 = pd.DataFrame(vMatrix1)
            dfv1.to_csv(f"{WsFileName}{timestamp}_vMatrix1.csv")
            if D > 2:
                dfv2 = pd.DataFrame(vMatrix2)
                dfv2.to_csv(f"{WsFileName}{timestamp}_vMatrix2.csv")

            print(f"[INFO] Solution saved at {time.strftime('%Y-%m-%d %H:%M:%S')} - Objective: {objective_value}")


status = GRB.INFEASIBLE #initial condition set for while infeasible loop 
ti = 0

while status == GRB.INFEASIBLE: 

    #Dimensions and constants 
    T = 45.0 + ti #total time in seconds 
    delT = 1.2 #seconds per iteration 
    N = round(T/delT) #iterations = N 
    S = 2 #sides of each dimension

    wingspan = 0.267 #wingspan of aircraft in thousands of feet

    A = 5 # number of formation aircraft
    D = 3 # dimensions (2 = 2-D, 3 = 3-D) 
    NI = 1 # number of non-cooperative intruders 
    intruder_degree = 0 # intruder approaching degree

    stringT = str(int(T)) 
    WsFileName = str(A)+"S"+ str(int(intruder_degree)) + "D" + stringT # 5 is the number of aircraft
    
    saveFileName = str(A)+"S"+ str(int(intruder_degree)) + "D"+ "0Avoidance.sol" # file name for saving solution file 

    #wind input 
    wind = np.zeros(D) 
    wind[0:2] = [0.000,0.000] # wind in 1k ft/sec: wind direction from [x direction E+/ W-, y direction N+ / S-] 
    if D > 2: 
        wind[2] = 0.000 # vertical wind from top (+) from bottom (-) 
    windi = wind*delT # wind

    #jetwash input (avoidance distance)
    jw = np.zeros(D) 
    jw[1] = 0.375*wingspan 
    if D > 2: 
        jw[2] = 0.375*wingspan
    js = 0.007 # sink rate of jetwash and vortices in 1k*ft/s (300ft/min sink rate or btw 5-7 ft/sec) 
    jsi = js*delT # sink rate of jetwash and vortices in 1k*ft/iteration 
    
    wns = np.zeros(D) # wind (horizontal x and y) and sink rate (z) for jetwash 
    wns = windi # wind per iteration input from top of code 
    if D > 2: 
        wns[2] += jsi # sink rate input above

    # VELOCITY AND ACCELERATION CONSTRAINTS HERE 
    K = 16 # number of the edges for polygon as per smoothness criteria 
    vmax = 0.835 # maximum velocity 1k*ft/s 
    vmin = 0.785 # minimum velocity 1k*ft/s 
    umax = 0.015 # maximum acceleration 1k*ft/s^2 
    vzmax = 0.020 # maximum vertical velocity (climb rate) 1k*ft/s 
    vzmin = -0.040 # maximum descent rate 1k*ft/s 
    uzmax = 0.015 # maximum acceleration in vertical 1k*ft/s^2 
    
    startDist = 1.335 # longitudinal (along-course) initial distance between aircraft 
    climbWeight = 10 # weight for acceleration/climb slack variable
  
    # plot settings 
    plotStop = N # stop the plot at a specific iteration to check aircraft/jetwash impingement - full plot is N
    jetwashTrack = 1 # set jetwash track to 0 (off) or 1 (on) for formation members on the 3-D plot 
    
    # Initialization parameters 
    x0 = np.zeros((D, A), dtype=float)

    # Generated inital points 
    offset = np.zeros(A) 
    sideCount = 0 
    for ac in range(A): # setting aircraft spacing required from courseline for each side 
            offset[ac] = sideCount 
            if (ac % 2) == 0: 
                sideCount +=1
    for ac in range(A): # join initial points for all aircraft except intruders 
                if ac == 0: 
                    sign = 0 
                elif ac !=0 and (ac % 2) == 0: 
                    sign = 1 
                else: 
                    sign = -1
                x0[0, ac] = startDist*(A-ac) 
                x0[1, ac] = sign*(offset[ac]*0.9*wingspan) 
                if D > 2: 
                    x0[2, ac] = 0.0


    #initial velocities and accelerations 
    vinit = 0.810 #ft/s 
    v0 = np.zeros((D, A)) 
    v0[0,] += vinit #initial velocities heading 090 
    u0 = np.zeros((D, A)) #initial acceleration zero

    # final parameters 
    finalCourseWidth = np.zeros((D, A, S), dtype=float) #aircraft need to be on final altitude of z=0 
    for ac in range(A): 
        if ac == 0: 
            finalCourseWidth[1, ac, 0] = 0 
            finalCourseWidth[1, ac, 1] = 0 
        else: 
            finalCourseWidth[1, ac, 0] = (0.9*wingspan)*offset[ac] #right side final course requirements for formation 
            finalCourseWidth[1, ac, 1] = (0.9*wingspan)*offset[ac] #left side final course requirements for formation 
        if D > 2: 
            finalCourseWidth[2, ac, 0] = 0.0 #bottom side final course requirements for formation 
            finalCourseWidth[2, ac, 1] = 0.0 #top side final course requirements for formation 
             

    # final velocity and acceleration constraints 
    vfLock = 5 # final velocity constraint active(# of iterations)/inactive(0) 
    ufLock = 5 # final acceleration constraint active(# of iterations)/inactive(0) 
    
    vf = v0 # specified final velocity 
    uf = np.zeros((D, A)) # specified final acceleration 

    intruder_radians = math.radians(intruder_degree)
    # non-intruder position, velocity 
    y = np.zeros((D, N, NI))
    vy = np.zeros((D,NI))
    vy[0,0] = -1*math.cos(intruder_radians)*0.81
    vy[1,0] = -1*math.sin(intruder_radians)*0.81
    vy[2,0] = 0.0
 
    for iter in range(N):
        y[0,iter,0] = (20.5/math.cos(intruder_radians/2))*(1+math.cos(intruder_radians)) + iter*delT*vy[0,0]
        y[1,iter,0] = (20.5/math.cos(intruder_radians/2))*math.sin(intruder_radians) + iter*delT*vy[1,0]
        if D>2:
            y[2,iter,0] = 0.0 + iter*delT*vy[2,0]

    # safe distance with intruder
    safetyDist = np.zeros((D), dtype=float) 
    for ni in range(NI): 
        safetyDist[0] = 2
        safetyDist[1] = 2 
        if D > 2: 
            safetyDist[2] = 1

    stringN = str(int(N)) 
    if ti == 0: 
        pklFileName = WsFileName+'.pkl' 
        dill.dump_session(pklFileName)   
    else: WsFileName = WsFileName+"T"+stringT 


    # model setup 
    m = gp.Model()
    
    # Create variables 
    x = m.addVars(D, N, A, lb=float('-inf'), ub=float('inf'), vtype=GRB.CONTINUOUS) # 3-D includes x y z 
    v = m.addVars(D, N, A, lb=float('-inf'), ub=float('inf'), vtype=GRB.CONTINUOUS) # 3-D includes vx vy vz 
    u = m.addVars(D, N, A, lb=float('-inf'), ub=float('inf'), vtype=GRB.CONTINUOUS) # 3-D includes ux uy uz 
    
    # Create slack variables (p for position, w for acceleration)
    p = m.addVars(D, lb=float('-inf'), ub=float('inf'), vtype=GRB.CONTINUOUS) # slack variable final breakUp minimization of distance 
    w = m.addVars(D, N, A, lb=float('-inf'), ub=float('inf'), vtype=GRB.CONTINUOUS) # slack variable control 
    t = m.addVars(D, N, A, lb=float('-inf'), ub=float('inf'), vtype=GRB.CONTINUOUS) # slack variable time for intruders 

    # initialization constraints 
    for ac in range(A): #constraints for formation 
        for d in range(D): 
            m.addConstr((x[d, 0, ac] == x0[d, ac])) 
            m.addConstr((v[d, 0, ac] == v0[d, ac])) 
            m.addConstr((u[d, 0, ac] >= u0[d, ac])) 

    # final course contraints 
    P = m.addVars(A, A, vtype=GRB.BINARY, name="P") # binary decision matrix
    for i in range(A):
        m.addConstr(sum(P[i, j] for j in range(A)) == 1)
    for j in range(A):
        m.addConstr(sum(P[i, j] for i in range(A)) == 1)
    for ac in range(A): #all aircraft are checked for final course constraints defined above in formation and intruder sections 
        for d in range(1, D):  
            for s in range(S): 
                if (s % 2) == 0: 
                    sign = 1 
                else: 
                    sign = -1 
                m.addConstr(-sign * x[d, N-1, ac] <= gp.quicksum(P[ac, ac2] * finalCourseWidth[d, ac2, s] for ac2 in range(A))) 
                

    if vfLock != 1: #constant set velocity constraint at end of optimization 
        for ac in range(A): 
            for d in range(D): 
                for n in range(N-vfLock,N): 
                    m.addConstr((v[d, n, ac] == vf[d, ac])) 
    
    if ufLock != 1: #constant set acceleration constraint at end of optimization 
        for ac in range(A):
            for d in range(D): 
                for n in range(N-ufLock,N): 
                    m.addConstr((u[d, n, ac] == uf[d, ac])) 

    # slack variable control (acceleration) constraints 
    for ac in range(A): 
        for d in range (D): 
            m.addConstrs((u[d, i, ac] <= w[d, i, ac] for i in range(N))) 
            m.addConstrs((-u[d, i, ac] <= w[d, i, ac] for i in range(N))) 
            if d == 2: 
                m.addConstrs((climbWeight*v[d, i, ac] <= w[d, i, ac] for i in range(N))) 
            m.addConstrs((0 <= w[d, i, ac] for i in range(N))) 

    # slack variable distance from final requirements constraints/time penalty 
    for ac in range(A): # cooperative intruder must be within finalCourseWidth as well unless this is A-I 
        for d in range (1,D): 
            m.addConstrs((x[d, i, ac] - (finalCourseWidth[d, ac, 1]) <= t[d, i, ac] for i in range(N))) #position at iteration i is less than finalCourseWidth 
            m.addConstrs((-x[d, i, ac] - (finalCourseWidth[d, ac, 0]) <= t[d, i, ac] for i in range(N))) #position at iteration i is greater than -finalCourseWidth 
            m.addConstrs((t[d, i, ac] >= 0 for i in range(N)))  

    #slack variable for drag
    delta = m.addVars(N,A,vtype=GRB.BINARY, name="delta")  # leading flight(delta=1),others(delta=0)
    ff = m.addVars(N,A,A, vtype=GRB.BINARY, name="ff")  # forward flight(fo=1),others(fo=0)
    md = m.addVars(D,N,A, lb=float('-inf'), ub=float('inf'),vtype=GRB.CONTINUOUS, name="md")  #distance between forward flight and itself

    fuel = m.addVars(D,N,A, lb=float('-inf'), ub=float('inf'),vtype=GRB.CONTINUOUS, name="fuel")
    Mc1 = m.addVars(N,A,vtype=GRB.CONTINUOUS, lb=0, ub=1, name="Mc1") # fuel[0,i,ac] * fuel[2,i,ac]
    Mc2 = m.addVars(N,A,vtype=GRB.CONTINUOUS, lb=0, ub=1, name="Mc2") # fuel[0,i,ac] * fuel[1,i,ac] *fuel[2,i,ac]


    M = 1000.000
    #find the leading aircraft in every interval
    for i in range(N):
        for ac1 in range(A):
            for ac2 in range(A):
                if ac1 != ac2:
                    m.addConstr(x[0,i,ac1]>=x[0,i,ac2]+(delta[i,ac1]-1)*M)
        m.addConstr(gp.quicksum(delta[i,ac1] for ac1 in range(A))==1)

    #find the forward flight
    for i in range(N):
        for ac1 in range(A):
            for ac2 in range(A):
                if ac1 !=ac2:
                    m.addConstr(md[0,i,ac1]<=(x[0,i,ac2]-x[0,i,ac1])+M*(1-ff[i,ac1,ac2])+M*delta[i,ac1])
                    m.addConstr(md[0,i,ac1]>=0)
            m.addConstr(gp.quicksum(ff[i,ac1,ac2] for ac2 in range(A) if ac2 != ac1) == 1-delta[i,ac1])

    #calculate the distance between the forword flight and itself
    for i in range(N):
        for ac1 in range(A):
            m.addConstr(md[1,i,ac1]>=gp.quicksum((x[1,i,ac2]-x[1,i,ac1])*ff[i,ac1,ac2] for ac2 in range(A) if ac2 != ac1))
            m.addConstr(md[1,i,ac1]>=gp.quicksum((-x[1,i,ac2]+x[1,i,ac1])*ff[i,ac1,ac2] for ac2 in range(A) if ac2 != ac1))
            m.addConstr(md[2,i,ac1]==gp.quicksum((-x[1,i,ac2]+x[1,i,ac1])*ff[i,ac1,ac2] for ac2 in range(A) if ac2 != ac1))
    
    #slack variable of fuel
    for i in range(N):
        for ac in range(A):
            m.addGenConstrPWL(md[0,i,ac], fuel[0,i,ac], [0, 0.005, startDist, 5*startDist], [0, 1, 1, 0.2])
            m.addGenConstrPWL(md[1,i,ac], fuel[1,i,ac], [0, 0.8*wingspan, wingspan,wingspan+0.01,100*wingspan], [0, 0.4, 0.4, 0.1, 0.1])
            m.addGenConstrPWL(md[2,i,ac], fuel[2,i,ac], [-100*wingspan, 0.075*wingspan, 0.075*wingspan+0.01,100*wingspan], [0, 0, 1, 1])
            #McCormick Slack fuel cost function
            m.addConstr(Mc1[i,ac] >= fuel[0,i,ac] + fuel[2,i,ac] - 1)
            m.addConstr(Mc1[i,ac] <= fuel[0,i,ac])
            m.addConstr(Mc1[i,ac] <= fuel[2,i,ac])
            m.addConstr(Mc1[i,ac] >= 0)
            m.addConstr(Mc2[i,ac] >= Mc1[i,ac] + fuel[1,i,ac] - 1)
            m.addConstr(Mc2[i,ac] <= Mc1[i,ac])
            m.addConstr(Mc2[i,ac] <= fuel[1,i,ac])
            m.addConstr(Mc2[i,ac] >= 0)
            m.addConstr(Mc2[i,ac] <= 0.4)
                    
    # model dynamics constraints x(i+1) = Ax(i) + Bu(i) 
    for ac in range(A): 
        for d in range(D): 
            m.addConstrs((x[d, i, ac] +delT * (v[d, i, ac] - wind[d]) == x[d, i+1, ac] for i in range(N-1))) 
            m.addConstrs((v[d, i, ac] +delT * u[d, i, ac] == v[d, i+1, ac] for i in range(N-1))) 
        
    M = 1000.000 

    # velocity constraints
    ec = m.addVars(A, K, N, vtype=GRB.BINARY) #no difference in forward, lateral acceleration 
    for ac in range(A): 
        for k in range(K): #horizontal velocity and acceleration limits 
            m.addConstrs((v[0, i, ac]*math.sin((2*pi*k)/K) + v[1, i, ac]*math.cos((2*pi*k)/K) <= vmax for i in range(N))) 
            m.addConstrs((v[0, i, ac]*math.sin((2*pi*k)/K) + v[1, i, ac]*math.cos((2*pi*k)/K) >= vmin - M*ec[ac, k, i] for i in range(N))) 
            m.addConstrs((u[0, i, ac]*math.sin((2*pi*k)/K) + u[1, i, ac]*math.cos((2*pi*k)/K) <= umax for i in range(N))) 
            if D > 2: 
                m.addConstrs((u[0, i, ac]*math.sin((2*pi*k)/K) + u[1, i, ac]*math.cos((2*pi*k)/K) + v[2, i, ac] <= umax for i in range(N))) #cannot climb and turn well + v[2, i, ac] 
        for i in range(N):
            m.addConstr((gp.quicksum([ec[ac, k, i] for k in range(K)]) <= K-1 ))
        if D>2: #vertical velocity and acceleration limits 
            m.addConstrs((v[2, i, ac] <= vzmax for i in range(N))) 
            m.addConstrs((v[2, i, ac] >= vzmin for i in range(N))) 
            m.addConstrs((u[2, i, ac] <= uzmax for i in range(N))) 
            m.addConstrs((u[2, i, ac] >= -uzmax for i in range(N))) 

    # collision avoidance of formation members 
    bR = 0.500 # Bubble radius in ft one aicraft (1000' between two) due to plotting 
    sepReq = np.ones((D, A)) 
    sepReq[:, :] = 0.5 #separation requirements for formation members and exit/escapers 
    # binary slack variables for collision avoidance 
    M = 1000.000 #arbitrary large number 
    e1 = m.addVars(S, D, N, A, A, vtype=GRB.BINARY)
    for ac in range(A): # every vehicle 
        for ac2 in range(A): # every vehicle 
            if ac != ac2: # other vehicle 
                for d in range(D): #dimensions 
                    for s in range(S): #sides, check both sides 
                        if (s % 2) == 0: #if side is even, then check (position of ac - position of ac2) 
                            sign = 1 #if side is odd, then check (position of ac2 - position of ac) 
                        else: 
                            sign = -1 
                        m.addConstrs((sign*x[d, i, ac] - sign*x[d, i, ac2] >= (sepReq[d, ac] + sepReq[d, ac2]) - M*e1[s, d, i, ac, ac2] for i in range(1,N)))       
                m.addConstrs((gp.quicksum([e1[0, d, i, ac, ac2] + e1[1, d, i, ac, ac2] for d in range(D)]) <= (S*D-1) for i in range(1,N))) 

    #wake turbulence variables 
    e2 = m.addVars(S, D, N, N, A, A, vtype=GRB.BINARY) 
    #wake turbulence avoidance 
    for i in range(N): 
        for ac in range(A): # every vehicle 
            for ac2 in range(A): # every vehicle 
                if ac != ac2: # not comparing an aircraft against itself 
                    for d in range(D): # every dimension 
                        for s in range(S): # both sides of the dimension 
                            if (s % 2) == 0: 
                                sign = 1 
                            else:
                                sign = -1 
                            if d == 0: #jw needs to be long enough in velocity direction to make sure ac2 can't hide in digital "holes" between iterations: fixed in avoidance code based on vx 
                                m.addConstrs((sign*x[d, i, ac] - sign*(x[d, i2, ac2]-wns[d]*(i-i2)) >= (v[d,i2,ac2]*delT) - M*e2[s, d, i, i2, ac, ac2] for i2 in range(i))) 
                            else: #jw in y and z direction based only on jw wind and sink - only works for x primary velocity direction 
                                m.addConstrs((sign*x[d, i, ac] - sign*(x[d, i2, ac2]-wns[d]*(i-i2)) >= jw[d] - M*e2[s, d, i, i2, ac, ac2] for i2 in range(i))) 
                                    #add all binary variables and ensure that one distance is outside jetwash box 
                    m.addConstrs((gp.quicksum([e2[0, d, i, i2, ac, ac2] + e2[1, d, i, i2, ac, ac2] for d in range(D)]) <= (S*D-1) for i2 in range(i)))

    #wake turbulence variables of non intruder
    e3 = m.addVars(S, D, N, N, A, NI, vtype=GRB.BINARY) 
    #wake turbulence avoidance 
    for i in range(N): 
        for ni in range(NI): # every vehicle 
            for ac in range(A): # every vehicle 
                for d in range(D): # every dimension 
                    for s in range(S): # both sides of the dimension 
                        if (s % 2) == 0: 
                            sign = 1 
                        else:
                            sign = -1 
                        if d == 0: #jw needs to be long enough in velocity direction to make sure ac2 can't hide in digital "holes" between iterations: fixed in avoidance code based on vx 
                            m.addConstrs((sign*x[d, i, ac] - sign*(y[d, i2, ni]-wns[d]*(i-i2)) >= (vy[d,ni]*delT) - M*e3[s, d, i, i2, ac, ni] for i2 in range(i))) 
                        else: #jw in y and z direction based only on jw wind and sink - only works for x primary velocity direction 
                            m.addConstrs((sign*x[d, i, ac] - sign*(y[d, i2, ni]-wns[d]*(i-i2)) >= jw[d] - M*e3[s, d, i, i2, ac, ni] for i2 in range(i))) 
                                #add all binary variables and ensure that one distance is outside jetwash box 
                m.addConstrs((gp.quicksum([e3[0, d, i, i2, ac, ni] + e3[1, d, i, i2, ac,ni] for d in range(D)]) <= (S*D-1) for i2 in range(i)))


    # non-intruder avoidence constraints
    # binary slack variables for collision avoidance  
    z1 = m.addVars(D, N, A, NI, vtype=GRB.BINARY)
    z2 = m.addVars(D, N, A, NI, vtype=GRB.BINARY)
    for ac in range(A): # every vehicle 
        for ni in range(NI): # every vehicle 
                m.addConstrs((x[0, i, ac] - y[0, i, ni] >= safetyDist[0] - M*z1[0, i, ac, ni] for i in range(1,N)))
                m.addConstrs((-x[0, i, ac] + y[0, i, ni] >= safetyDist[0] - M*z2[0, i, ac, ni] for i in range(1,N))) 
                m.addConstrs((x[1, i, ac] - y[1, i, ni] >= safetyDist[1] - M*z1[1, i, ac, ni] for i in range(1,N)))
                m.addConstrs((-x[1, i, ac] + y[1, i, ni] >= safetyDist[1] - M*z2[1, i, ac, ni] for i in range(1,N)))  
                m.addConstrs((x[2, i, ac] - y[2, i, ni] >= safetyDist[2] - M*z1[2, i, ac, ni] for i in range(1,N)))
                m.addConstrs((-x[2, i, ac] + y[2, i, ni] >= safetyDist[2] - M*z2[2, i, ac, ni] for i in range(1,N)))       
                m.addConstrs((gp.quicksum([z1[d, i, ac, ni] + z2[d, i, ac, ni] for d in range(D)]) <= 5 for i in range(1,N))) 


    # cost function variables 
    ru = np.ones(D) # control useage weight 
    rt = np.ones(D) # time weight 
    rf = 1 # fuel weight
    # Cost function 
    J_cost1 = gp.LinExpr() 
    J_cost2 = gp.LinExpr() 
    J_cost3 = gp.LinExpr() 

    for i in range(N):
        for ac in range(A): 
            for d in range(1,D): 
                J_cost2 += (rt[d]*t[d, i, ac])
            for d in range(D): 
                J_cost1 += (ru[d]*w[d, i, ac])
            J_cost3 += rf*(1-Mc2[i,ac])
    
    
    J_cost = J_cost1 + J_cost2 + J_cost3

    m.Params.DualReductions = 0 #allows gurobi to discern between infeasible and unbounded in output message 
    #m.Params.PoolSearchMode = 2 #tells optimizer to search for n best solutions n = PoolSolutions 
    #m.Params.PoolSolutions = 5 #number of best solutions to search for 
    # m.Params.MIPGap = 0.02
    #m.Params.PoolGap = 10 #maximum objective gap between worst and best solutions 
    #m.Params.DegenMoves = 0
    m.Params.Presolve = -1
    m.Params.Threads = 64
    m.Params.Method = -1
    m.Params.MIPFocus = 1 
    #m.Params.Cuts = 2 
    
    # set objective function and optimizing 
    m.setObjective(J_cost, GRB.MINIMIZE) 

    m.optimize(my_callback) 
    print("T = ", T) 
    ti += 5 
    status = m.Status 
    
    if m.status == GRB.OPTIMAL:
        print("found solutions successfully")
        if "saveFileName" in globals(): 
            m.write(saveFileName) 

#set different solution numbers for saving and plotting 
    SolCount = m.SolCount #SolCount parameter contains the number of feasibile solutions found 
    prevObj = -3.1
    for sN in range(SolCount): #loop for plotting successive solutions 
        m.Params.SolutionNumber = sN #set solution number (name) for plotting 
        if m.PoolObjVal <= prevObj+3: 
            continue 
        
        prevObj = m.PoolObjVal 

        sNum = str(sN) 
        Ob = str(int(m.PoolObjVal)) 
        xMatrix0 = np.zeros((N, A)) 
        for i in range(N): 
            for ac in range(A):
                xMatrix0[i, ac] = x[0, i, ac].Xn 
        xMatrix1 = np.zeros((N, A)) 
        for i in range(N): 
            for ac in range(A): 
                xMatrix1[i, ac] = x[1, i, ac].Xn 
        if D > 2: 
            xMatrix2 = np.zeros((N, A)) 
            for i in range(N): 
                for ac in range(A): 
                    xMatrix2[i, ac] = x[2, i, ac].Xn

        dfx0 = pd.DataFrame(xMatrix0) 
        dfx0.to_csv(WsFileName+sNum+Ob+'xMatrix0.csv') 
        dfx1 = pd.DataFrame(xMatrix1) 
        dfx1.to_csv(WsFileName+sNum+Ob+'xMatrix1.csv') 
        if D > 2: 
            dfx2 = pd.DataFrame(xMatrix2) 
            dfx2.to_csv(WsFileName+sNum+Ob+'xMatrix2.csv') 
            
        vMatrix0 = np.zeros((N, A)) 
        for i in range(N): 
            for ac in range(A): 
                vMatrix0[i, ac] = v[0, i, ac].Xn 
        vMatrix1 = np.zeros((N, A)) 
        for i in range(N): 
            for ac in range(A): 
                vMatrix1[i, ac] = v[1, i, ac].Xn 
        if D > 2: 
            vMatrix2 = np.zeros((N, A)) 
            for i in range(N): 
                for ac in range(A): 
                    vMatrix2[i, ac] = v[2, i, ac].Xn

        dfv0 = pd.DataFrame(vMatrix0) 
        dfv0.to_csv(WsFileName+sNum+Ob+'vMatrix0.csv')
        dfv1 = pd.DataFrame(vMatrix1) 
        dfv1.to_csv(WsFileName+sNum+Ob+'vMatrix1.csv') 
        if D > 2: 
            dfv2 = pd.DataFrame(vMatrix2) 
            dfv2.to_csv(WsFileName+sNum+Ob+'vMatrix2.csv')
