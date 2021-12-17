import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import project as p
import benders

# # suppress all warnings
# import warnings
# warnings.filterwarnings("ignore")

# import os
# # print("Current Working Directory " , os.getcwd())
# os.chdir("/Users/zhuxiaoy/Desktop/MAE&IOE&Data Science/(3) 2021Fall/IOE 591 Special Topics on Transportation System Optimization/Project/Code")

# define decision variable for kmeans
class DecisionVar:
    def __init__(self):
        self.x = {}
        self.y = {}
        self.z = {}
        #self.u = {}
        self.s = {}
        self.O = {}
        self.I = {}
        self.W= {}


def assign_PatientDepotVehicle(max_iter=500):
    # parameter: max_iter
    # this is the max iteration kmeans will run for each depot
    ############### assign patients to depots ######################
    # load in depots and patients information and covert it to dataframes
    depots = sets.depot
    patients = sets.patient
    depot_df = pd.DataFrame(columns=['name', 'locX', 'locY'])
    patient_df = pd.DataFrame(columns=['name', 'locX', 'locY', 'tStart', 'tEnd'])
    for d in depots:
        depot_df = depot_df.append({'name': d.name, 'locX': d.loc[0], 'locY': d.loc[1]}, ignore_index=True)
    for pa in patients:
        patient_df = patient_df.append({'name': pa.name, 'locX': pa.loc[0], 'locY': pa.loc[1], 'tStart': pa.tStart, 'tEnd': pa.tEnd}, ignore_index=True)

    # standardize data
    stand_D = StandardScaler().fit_transform(depot_df[['locX', 'locY']])
    stand_P = StandardScaler().fit_transform(patient_df[['locX', 'locY']])
    
    # use Kmeans to cluster patients (n_cluster = depot number)    
    kmeans = KMeans(n_clusters=len(depots), max_iter=max_iter) 
    kmeans.fit(stand_P)

    ## after clustering, find the nearest depot for each centroid
    label_depot = {} # key is kmeans center, value is depot
    left_D = stand_D.copy()
    #for ind, c in pd.DataFrame(kmeans.cluster_centers_).drop([2, 3],axis=1).iterrows():
    for ind, c in enumerate(kmeans.cluster_centers_):
        min_dist = min([np.linalg.norm(d - c) for d in left_D])
        label_depot[ind] = [ind for ind, d in enumerate(stand_D) if np.linalg.norm(d - c) == min_dist][0]
        left_D = np.delete(left_D, [np.linalg.norm(d - c) for d in left_D].index(min_dist), axis = 0)
    ## assign patients to depots
    patient_df['depot'] = [label_depot[l] for l in kmeans.labels_] ## depot number

    ########################################## assign vehicles to depots ########################################
    # load in vehicles information and covert it to dataframes
    vehicles = sets.vehicle
    vehicle_df = pd.DataFrame(columns=['name', 'cap', 'totOprTime', 'oprCost'])
    for v in vehicles:
        vehicle_df = vehicle_df.append({'name': v.name, 'cap': v.cap, 'totOprTime': v.totOprTime, 'oprCost': v.oprCost}, ignore_index=True)

    # sort depots based on assigned patients in a decreasing order 
    depots_order = patient_df.groupby('depot').count().sort_values(['name'], ascending = False).index.tolist()
    # sort vehicles based on the ratio of operation cost to capacity in an increasing order
    vehicle_df = vehicle_df.loc[(vehicle_df['oprCost'].astype('float') / vehicle_df['cap'].astype('float')).sort_values().index]
    
    # for each iteration, assign vehicles to depots in the above order. Stop until all depots complete assignment
    vehi_left = vehicle_df.copy()
    pati_left = patient_df.copy()
    veh_depot = {} # key is vehicle, value is depot
    while not pati_left.empty:
        for d in depots_order:
            vehi_capa = vehi_left.loc[0, 'cap'] # first vehicle
            depot_pati = pati_left[pati_left['depot'] == d].reset_index(drop=True)
            if not depot_pati.empty:
                veh_depot[vehi_left.loc[0, 'name']] = d
                pati_served_name = depot_pati[:vehi_capa]['name']
                pati_left = pati_left.drop(list(pati_left[pati_left['name'].isin(pati_served_name)].index), axis=0).reset_index(drop=True)
                vehi_left = vehi_left.drop([0], axis=0).reset_index(drop=True)

    for v in veh_depot.keys():
        decisionVar.x[v] = 1
        decisionVar.z[v, depot_df.loc[veh_depot[v], 'name']] = 1
        
    ######################################### assign patients to vehicles #########################################
    routes = {} # key is vehicle, value is patient df
    for d in depot_df.index:
        allPatients = patient_df[patient_df.depot == d].reset_index(drop=True)
        allVehicles = [v for v,de in veh_depot.items() if de == d]
        if len(allVehicles) > 1:
            for veh in allVehicles:
                sort_veh = vehicle_df.loc[vehicle_df['name'].isin(allVehicles)].sort_values(['cap']).reset_index(drop=True)
                maxCap = sort_veh.loc[len(sort_veh)-1, 'cap']
                #minCap = sort_veh.loc[0, 'cap']

            Depotkmeans = KMeans(n_clusters=maxCap, max_iter=max_iter) # do kmeans using the max capacity
            # standardize data
            stand_allP = StandardScaler().fit_transform(allPatients[['tStart', 'tEnd']])
            Depotkmeans.fit(stand_allP)
            allPatients.loc[:,'label'] = Depotkmeans.labels_
            for ind, veh in sort_veh.iterrows(): # assign vehicles with the least capcaity first
                route_df = pd.DataFrame(columns = allPatients.columns)
                labels = allPatients.label.unique()
                labels.sort()
                for i in labels:
                    route_df = route_df.append(allPatients[allPatients['label'] == i].reset_index(drop=True).iloc[0])
                    allPatients = allPatients.drop(allPatients[allPatients['name'].isin(route_df['name'])].index, axis=0).reset_index(drop=True)
                while len(route_df) < veh['cap'] and not allPatients.empty:
                    capa_left = veh['cap'] - len(labels)
                    labels = allPatients.label.unique()
                    labels.sort()
                    for i in labels[:capa_left]:
                        route_df = route_df.append(allPatients[allPatients['label'] == i].reset_index(drop=True).iloc[0])
                        allPatients = allPatients.drop(allPatients[allPatients['name'].isin(route_df['name'])].index, axis=0).reset_index(drop=True)
                route_df = route_df.sort_values(by=['tStart']).reset_index(drop=True)
                routes[veh['name']] = route_df

                for ind, pa in route_df.iterrows():
                    decisionVar.y[pa['name'], veh['name']] = 1 
                    decisionVar.s[depot_df.loc[route_df.depot[0],'name'], route_df.loc[0,'name'], veh['name']] = 1 
                    if ind > 0:
                        decisionVar.s[route_df.loc[ind-1,'name'], pa['name'], veh['name']] = 1 
                decisionVar.s[route_df.loc[len(route_df)-1,'name'], depot_df.loc[route_df.depot[0],'name'], veh['name']] = 1 

        else:
            veh_route_df = allPatients.sort_values(by = ['tStart']).reset_index(drop=True)
            routes[allVehicles[0]] = veh_route_df
            allPatients = allPatients.drop(allPatients[allPatients['name'].isin(veh_route_df['name'])].index, axis=0).reset_index(drop=True)


            for ind, pa in veh_route_df.iterrows():
                decisionVar.y[pa['name'], allVehicles[0]] = 1 
                decisionVar.s[depot_df.loc[veh_route_df.depot[0],'name'], veh_route_df.loc[0,'name'], allVehicles[0]] = 1 
                if ind > 0:
                    decisionVar.s[veh_route_df.loc[ind-1,'name'], pa['name'], allVehicles[0]] = 1 
            decisionVar.s[veh_route_df.loc[len(veh_route_df)-1,'name'], depot_df.loc[veh_route_df.depot[0],'name'], allVehicles[0]] = 1 
        print("Whether all patients assigned to the depot {} has a vehicle assigned: {}".format(depot_df.loc[d, 'name'], allPatients.empty))

    return routes

# an auxiliary function to help calculate idle time, waiting time, overtime
def cal_IWReal(EA, wholeTime):
    if wholeTime <= EA:
        preI = EA - wholeTime
        curW = 0
        RA = EA
    elif wholeTime >= EA:
        preI = 0
        curW = wholeTime - EA
        RA = wholeTime
    return preI, curW, RA


# calculate idle, waiting, overtime
def cal_IWO(routes):
    for w in sets.scenario:
        for veh, route in routes.items():
            depotName = [pair[1] for pair in decisionVar.z.keys() if pair[0] == veh][0]
            RA = 0 # to store real arrival time for each patient. start from 0. 
            for position, patient in route.iterrows(): # iterate over patients in each route
                EA = route.loc[position, 'tStart']
                LA = route.loc[position, 'tEnd']
                if position == 0: # if it's the first patient
                    wholeTime = RA + para.travelT[depotName, patient['name'], w]
                    decisionVar.I[depotName, veh, w], decisionVar.W[patient['name'], veh, w], RA = cal_IWReal(EA, wholeTime)
                else:
                    wholeTime = RA + para.svcT[route.loc[position-1, 'name'],w] + para.travelT[route.loc[position-1, 'name'], patient['name'], w]
                    decisionVar.I[route.loc[position-1, 'name'], veh, w], decisionVar.W[patient['name'], veh, w], RA = cal_IWReal(EA, wholeTime)
            # after finishing the for loop of one route dataframe, calculate the overtime
            # idle time for the last patient should be zero
            decisionVar.O[veh, w] = max(RA + 0 + para.svcT[route.loc[len(route)-1, 'name'],w] + para.travelT[route.loc[len(route)-1, 'name'], depotName, w] - sets.vehicle[veh].totOprTime , 0)
    pass

# calculate total cost
def cal_ToTCost():
    ToTOperCost = 0
    for veh in sets.K:
        try:
            ToTOperCost = ToTOperCost + para.oprCost[veh]*decisionVar.x[veh]
        except:
            continue

    ToTtravCost = 0
    for w in sets.scenario:
        for event in decisionVar.s:
            i = event[0]
            j = event[1]
            veh = event[2]
            ToTtravCost = ToTtravCost + sets.p[w]*(para.traCost[i,j] * para.travelT[i,j,w] * decisionVar.s[i,j,veh])

    ToTIWO = 0
    for w in sets.scenario:
        for event in decisionVar.W:
            i = event[0]
            veh = event[1]
            ToTIWO = ToTIWO + sets.p[w]*para.waitingPenalty*decisionVar.W[i,veh,w] 
        
        for event in decisionVar.I:
            i = event[0]
            veh = event[1]
            ToTIWO = ToTIWO + sets.p[w]*para.idlePenalty*decisionVar.I[i,veh,w]
            
        for event in decisionVar.O:
            veh = event[0]
            ToTIWO = ToTIWO + sets.p[w]*para.overtimePenalty*decisionVar.O[veh,w]

    ToTCost = ToTOperCost + ToTtravCost + ToTIWO
    return ToTCost

# print results to files
def printResult(routes, ToTCost, max_iter=500):
    # Assignments
    vehicles = sets.vehicle
    vehicle_df = pd.DataFrame(columns=['name', 'cap', 'totOprTime', 'oprCost'])
    for v in vehicles:
        vehicle_df = vehicle_df.append({'name': v.name, 'cap': v.cap, 'totOprTime': v.totOprTime, 'oprCost': v.oprCost}, ignore_index=True)
    AllVehicles = vehicle_df['name'].tolist()
    Vehi_used = list(decisionVar.x.keys())
    for k in AllVehicles:
        if k not in Vehi_used:
            vehStr = "Vehicle {} is not used".format(k)
        else:
            depotName = [pair[1] for pair in decisionVar.z.keys() if pair[0] == k][0]
            vehStr = "Vehicle {} assigned to depot {}.".format(k, depotName)
        print(vehStr, file=sets.f)
    # Routing
    print("",file=sets.f)
    for k in Vehi_used:
        depotName = [pair[1] for pair in decisionVar.z.keys() if pair[0] == k][0]
        routeStr = str(depotName)
        route = routes[k]
        for ind, patient in route.iterrows():
            routeStr += " -> patient{} ".format(int(patient['name']))
                    
        routeStr += " -> {} ".format(depotName)
                    
        print("Vehicle {}'s route: {}".format(k, routeStr),file=sets.f)
    
    # Objective Value
    print("",file=sets.f)
    print("Kmeans Iterations: {}".format(max_iter), file=sets.f)
    print("Objective Value: {:.4f}".format(ToTCost),file=sets.f)
    print("Average Idle Time: {:.2f}".format(sum(decisionVar.I.values())/len(decisionVar.I.values())),file=sets.f)
    print("Average Waiting Time: {:.2f}".format(sum(decisionVar.W.values())/len(decisionVar.W.values())),file=sets.f)
    print("Average Over Time: {:.2f}".format(sum(decisionVar.O.values())/len(decisionVar.O.values())),file=sets.f)

    
sets = p.Sets("({}-{})".format(5,20),10)
para = p.Parameters(sets)
decisionVar = DecisionVar()


p.printScen("Solving TSMILP Model using Benders' Decomposition",sets.f)
start_time = time.time()
m = benders.MasterProblem(para,sets)
m.optimize()
end_time = time.time()
p.printScen("time taken = "+str(end_time-start_time),sets.f)

# # p.printScen("Solving TSMILP Model",sets.f)
# # start_time = time.time()
# # m = p.TSMILP(sets,para)
# # m.optimize()
# # end_time = time.time()
# # p.printScen("time taken = "+str(end_time-start_time),sets.f)

p.printScen("Solving the problem using Kmeans Heuristic",sets.f)
start_time = time.time()
max_iter = 500
routes = assign_PatientDepotVehicle(max_iter = max_iter)
cal_IWO(routes)
ToTCost = cal_ToTCost()
printResult(routes, ToTCost)
end_time = time.time()
p.printScen("time taken = "+str(end_time-start_time) + 's',sets.f)

# for i in range(3,6):
#     for j in [10,20]:
#         sets = p.Sets("({}-{})".format(i,j),10)
#         para = p.Parameters(sets)
#         decisionVar = DecisionVar()


#         # p.printScen("Solving TSMILP Model using Benders' Decomposition",sets.f)
#         # start_time = time.time()
#         # m = benders.MasterProblem(para,sets)
#         # m.optimize()
#         # end_time = time.time()
#         # p.printScen("time taken = "+str(end_time-start_time),sets.f)

#         p.printScen("Solving TSMILP Model",sets.f)
#         start_time = time.time()
#         m = p.TSMILP(sets,para)
#         m.optimize()
#         end_time = time.time()
#         p.printScen("time taken = "+str(end_time-start_time),sets.f)


#         # p.printScen("Solving the problem using Kmeans Heuristic",sets.f)
#         # start_time = time.time()
#         # max_iter = 500
#         # routes = assign_PatientDepotVehicle(max_iter = max_iter)
#         # cal_IWO(routes)
#         # ToTCost = cal_ToTCost()
#         # printResult(routes, ToTCost)
#         # end_time = time.time()
#         # p.printScen("time taken = "+str(end_time-start_time) + 's',sets.f)
