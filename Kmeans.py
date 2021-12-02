import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# suppress all warnings
import warnings
warnings.filterwarnings("ignore")

import os
# print("Current Working Directory " , os.getcwd())
os.chdir("/Users/zhuxiaoy/Desktop/MAE&IOE&Data Science/(3) 2021Fall/IOE 591 Special Topics on Transportation System Optimization/Project/Code")

# write a function to assign patients to depots using kmeans
def assign_depots(depots, patients):
    # standardize data
    D = StandardScaler().fit_transform(depots[['locX', 'locY']] )
    standard_data = StandardScaler().fit_transform(patients[['X', 'Y']] )
    
    # let 5 depots to be our centroids
    centroids = D
    kmeans = KMeans(n_clusters=5, init=centroids, max_iter=1) # just run one k-Means iteration so that the centroids are not updated

    kmeans.fit(standard_data)
    return kmeans.labels_

# assign vechicles to depots
def VehicleAssign(vehicles, patients):
    routes = {}
    all_vehicles = list(vehicles.index)
    for i in range(5):
        depoti = patients[patients['depot'] == i].sort_values('EarlyArrival').reset_index(drop=True)
        for v in all_vehicles:
            routes_capa = vehicles.loc[v, 'Capacity']
            if not depoti.empty:
                routes[v] = depoti[:routes_capa]
                depoti = depoti.drop(list(routes[v].index)).reset_index()
                all_vehicles.remove(v)
            else:
                break
    print("Whether all depots complete vehicle assignment: {}.".format(depoti.empty))
    return routes

# track travelling variables including travel distance, travel cost
def calcul_travelDistCost(routes):
    # calculate travel distance for each route
    newRoutes = {}
    for i in routes.keys():
        routei = routes[i].copy() # get the information of each route
        d = routei.depot[0] # get the depot of the route (depot of all patients in the same route is the same)
        for position, patient in routei.iterrows(): # iterate over patients in each route
            if position == 0: # for the first patient
                depotX = depots.loc[d, 'locX'] # get the location of the depot
                depotY = depots.loc[d, 'locY']
                currX = routei.loc[position, 'X'] # get the location of the patient
                currY = routei.loc[position, 'Y']
                TravelDist = math.sqrt((currX-depotX)**2 + (currY-depotY)**2) # calculate the distance
                routei.loc[position, 'TravelDist'] = TravelDist 
            elif position == routei['Patient'].dropna().index[-1]: 
                # calculate the distance between the last patient and second last patient
                prevX = routei.loc[position-1, 'X']
                prevY = routei.loc[position-1, 'Y']
                currX = routei.loc[position, 'X']
                currY = routei.loc[position, 'Y']
                TravelDist = math.sqrt((currX-prevX)**2 + (currY-prevY)**2)
                routei.loc[position, 'TravelDist'] = TravelDist
                # calculate the distance between the last patient and the depot
                depotX = depots.loc[d, 'locX'] # get the location of the depot
                depotY = depots.loc[d, 'locY']
                currX = routei.loc[position, 'X']
                currY = routei.loc[position, 'Y']
                TravelBackDepot = math.sqrt((currX-depotX)**2 + (currY-depotY)**2)
            else:  # for the remaining patients
                prevX = routei.loc[position-1, 'X']
                prevY = routei.loc[position-1, 'Y']
                currX = routei.loc[position, 'X']
                currY = routei.loc[position, 'Y']
                TravelDist = math.sqrt((currX-prevX)**2 + (currY-prevY)**2)
                routei.loc[position, 'TravelDist'] = TravelDist
        
        # append the distance between the last patient and the depot to the dataframe
        routei.loc['EndDepot', 'TravelDist'] = TravelBackDepot
        
        
        # calculate travel cost
        routei['TravelCost'] = routei['TravelDist']*1
        # update the route information
        newRoutes[i] = routei 

    return newRoutes

def cal_IdleWaiting(routei, position, EA, LA, wholeTime, i):
    # i is scenario
    c3 = 'PreIdle' + str(i+1)
    c4 = 'curWating' + str(i+1)
    c5 = 'RealArrival' + str(i+1)
    if wholeTime <= EA:
        routei.loc[position, c3] = EA - wholeTime
        routei.loc[position, c4] = 0
        routei.loc[position, c5] = EA
    elif EA < wholeTime <= LA:
        routei.loc[position, c3] = 0
        routei.loc[position, c4] = 0
        routei.loc[position, c5] = wholeTime
    elif wholeTime > LA:
        routei.loc[position, c3] = 0
        routei.loc[position, c4] = wholeTime - LA
        routei.loc[position, c5] = wholeTime
    return routei


def scenario_Vars(routes, scen):
    # scenario-based variables: travel time, service time followed by idle time, waiting time, overtime
    # first only try one scenario
    finalRoutes = {}
    s = scen[0]
    
    for i in routes.keys():
        routei = routes[i].copy()
        #d = routei.depot[0] # get the depot of the route (depot of all patients in the same route is the same)
        # service time
        for position, patient in routei.iterrows(): # iterate over patients in each route
            c1 = 'ServTime' + str(0+1)
            try:
                routei.loc[position, c1] = s['service time'][patient.Patient - 1]
            except:
                continue
        # travel time
        c2 = 'travTime' + str(0+1)
        routei[c2] = routei['TravelDist']*6000/50
        
        # calculate idel, waiting time, over time
        for position, patient in routei.iterrows(): # iterate over patients in each route
            c3 = 'PreIdle' + str(0+1)
            c4 = 'curWating' + str(0+1)
            c5 = 'RealArrival' + str(0+1)
            EA = routei.loc[position, 'EarlyArrival']
            LA = routei.loc[position, 'LateArrival']
            if position != 'EndDepot':
                if position == 0: # for the first patient
                    wholeTime = 0 + routei.loc[position, c2] # staring time is zero
                    routei = cal_IdleWaiting(routei, position, EA, LA, wholeTime, 0)
                else: # for the remaining patients
                    wholeTime = routei.loc[position-1, c5] + routei.loc[position-1, c1] + routei.loc[position, c2]
                    routei = cal_IdleWaiting(routei, position, EA, LA, wholeTime, 0)
            else:
                lastPatient = routei['Patient'].dropna().index[-1]
                wholeTime = routei.loc[lastPatient, c5] + routei.loc[lastPatient, c1] + routei.loc['EndDepot', c2]
                routei.loc[position, c3] = 0 # when finish serving last patient, directly go back to the hospital
                Tk = vehicles.loc[i, 'Total Operation Time'] 
                if wholeTime <= Tk:
                    routei.loc[position, c4] = 0
                else:
                    routei.loc[position, c4] = wholeTime - Tk
                routei.loc[position, c5] = wholeTime
        
        finalRoutes[i] = routei

    return finalRoutes


def ouput_results(finalRoutes, vehicles):
    waitingPenalty = 2
    idlePenalty = 1
    overtimePenalty = 10
    vehi_used = list(finalRoutes.keys())
    # vehi_used.sort()
    print("Final Result:")
    print('-'*80)
    print("First {} vehicles are used.".format(len(vehi_used)))
    TotalCost = 0
    for i in vehi_used:
        TotalCost += vehicles.loc[i, 'Operation Cost'] + (finalRoutes[i].travTime1.sum()*1 + finalRoutes[i].PreIdle1.sum()*idlePenalty + finalRoutes[i].curWating1[:-1].sum()*waitingPenalty + finalRoutes[i].curWating1.tolist()[-1]*overtimePenalty)
    print("Total cost is {:.2f}.\n".format(TotalCost))
    print('-'*80)
    print('Routes for each vehicle:')
    print('-'*80)
    for i in vehi_used:
        routei = finalRoutes[i]
        print('Routes for Vehicle {} of depot {}:'.format(i+1, int(routei.depot[0])+1))
        AllRowind = routei.Patient.dropna().index.tolist()
        vehRoute = "depot" + str(int(routei.depot[0]))
        for r in AllRowind:
            vehRoute += " -> patient {} (dist={:.2f}, travelTime={:.2f}, serviceTime={})\n".format(int(routei.loc[r, 'Patient']), routei.loc[r, 'TravelDist'], routei.loc[r, 'travTime1'], routei.loc[r, 'ServTime1'])
        print(vehRoute)
    print('\n')
    print('-'*80)
    print('Idle, waiting, overtime for each vehicle route:')
    print('-'*80)
    for i in vehi_used:
        print('Time information for Vehicle {} of depot {}:\n'.format(i+1, int(routei.depot[0])+1))
        print("\tOvertime for vehicle {}: {:.2f}".format(i+1, finalRoutes[i].curWating1.tolist()[-1]))
        routei = finalRoutes[i]
        AllRowind = routei.Patient.dropna().index.tolist()
        vehstr = ""
        for r in AllRowind:
            vehstr += "\n\tPatient {}: Previous Idle Time = {}, Current Waiting Time = {}".format(int(routei.loc[r, 'Patient']), routei.loc[r, 'PreIdle1'], routei.loc[r, 'curWating1'])
        print(vehstr)
        print("\n")

# load in data
# can change
patients = pd.read_excel('data.xlsx', sheet_name = 'Patients')
vehicles = pd.read_excel('data.xlsx', sheet_name = 'Vehicles')
# fixed
depots = pd.read_excel('data.xlsx', sheet_name = 'Depots')

# assign patients to depots 
patients['depot'] = assign_depots(depots, patients)
# assign vechiles to all depots
allRoutes = VehicleAssign(vehicles, patients)
# calculate distance
newRoutes = calcul_travelDistCost(allRoutes)


# scenario setting
w = 4   # number of scenarios
scenarios = range(w)
p = [1/w for i in scenarios]    # probability of each scenario
scen = {}
# for s in scenarios:
#     scen[s] = 
scen[0] = {'speed':50, 'service time': patients['ServiceTime1']}
finalRoutes = scenario_Vars(newRoutes, scen)

# output results
ouput_results(finalRoutes, vehicles)
    
