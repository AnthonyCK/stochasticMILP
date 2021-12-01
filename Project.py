import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
from math import radians, cos, sin, asin, sqrt
import Benders 

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def TSMILP(para):
    try:
    # Create a new model
        model = gp.Model ("TSMILP")

    # Create variables
        x = model.addVars(K, vtype = GRB.BINARY, name = "x")                # operate veh k
        y = model.addVars(J, K, vtype = GRB.BINARY, name = "y")             # assign patient i to veh k
        z = model.addVars(K, D, vtype = GRB.BINARY, name = "z")             # assign veh k to depot d
        u = model.addVars(J, lb = 0.0, name = "u")                          # planned arrival time at i
        s = model.addVars(N, N, K, vtype = GRB.BINARY, name = "s")          # assign arc (i,j) to veh k

        O = model.addVars(K, scenarios, lb = 0.0, name = "O")                          # overtime of veh k
        I = model.addVars(N, K, scenarios, lb = 0.0, name = "I")                       # idletime at patient i of veh k
        W = model.addVars(N, K, scenarios, lb = 0.0, name = "W")                       # waitingtime at patient i of veh k
        model.update()
    # Set objective
        objective = gp.quicksum(para.oprCost[i]*x[i] for i in K)\
             + gp.quicksum(p[w]*(gp.quicksum(para.traCost[i,j]*para.travelT[i,j,w]*s[i,j,k] for i in N for j in N for k in K) + gp.quicksum(para.waitingPenalty*W[i,k,w] + para.idlePenalty*I[i,k,w] + para.overtimePenalty*O[k,w] for k in K for i in N)) for w in scenarios)
        model.setObjective(objective, GRB.MINIMIZE)

    # Add constraints
        model.addConstrs((gp.quicksum(z[k,d] for d in D)==1 for k in K), name = "assign-1")
        model.addConstrs((gp.quicksum(y[i,k] for k in K)==1 for i in J), name = "assign-2")
        # model.addConstrs((y[i,k] <=x[k] for i in J for k in K), name = "assign-3")
        model.addConstrs((gp.quicksum(s[i,j,k] for j in N) == y[i,k] for k in K for i in J), name="tour1")
        model.addConstrs((gp.quicksum(s[i,j,k] for i in N) == y[j,k] for k in K for j in J), name="tour2")
        model.addConstrs((gp.quicksum(s[d,i,k] for i in J)==z[k,d]*x[k] for k in K for d in D), name = "samedepot1")
        model.addConstrs((gp.quicksum(s[i,d,k] for i in J)==z[k,d]*x[k] for k in K for d in D), name = "samedepot2")
        model.addConstrs((gp.quicksum(y[i,k]*para.demand[i] for i in J)<=para.cap[k]*x[k] for k in K), name = "capacity")
        model.addConstrs((para.tStart[i]<=u[i] for i in J), name = "timewindow-1")
        model.addConstrs((u[i]<=para.tEnd[i] for i in J), name = "timewindow-2")
        M = {(i,j) : 24*60 + para.svcTMean[i] + para.traTMean[i,j] for i in J for j in J}
        model.addConstrs((u[j]>=u[i]+para.traTMean[i,j]+para.svcTMean[i]-M[i,j]*(1-gp.quicksum(s[i,j,k] for k in K)) for i in J for j in J), name = "planned_arri_time")

        # M = 60*60
        # model.addConstrs((W[j,k,w]<=travelT[i,j,w]+svcT[i,w]-u[j]+u[i]+W[i,k,w]+I[i,k,w] + M*(1-s[i,j,k]) for i in J for j in J for k in K for w in scenarios), name = "waitingtime-1")
        # model.addConstrs((O[k,w]<=travelT[i,d,w]+svcT[i,w]-totOprT[k]+u[i]+W[i,k,w]+I[i,k,w] + M*(1-s[i,d,k]) for i in J for d in D for k in K for w in scenarios), name = "overtime1")
        # model.addConstrs((W[i,k,w]<=travelT[d,i,w]-u[i]+I[d,k,w] + M*(1-s[d,i,k]) for i in J for d in D for k in K for w in scenarios), name = "waitingtime-2")
        # model.addConstrs((W[j,k,w]>=travelT[i,j,w]+svcT[i,w]-u[j]+u[i]+W[i,k,w]+I[i,k,w] - M*(1-s[i,j,k]) for i in J for j in J for k in K for w in scenarios), name = "waitingtime-3")
        # model.addConstrs((O[k,w]>=travelT[i,d,w]+svcT[i,w]-totOprT[k]+u[i]+W[i,k,w]+I[i,k,w] - M*(1-s[i,d,k]) for i in J for d in D for k in K for w in scenarios), name = "overtime2")
        # model.addConstrs((W[i,k,w]>=travelT[d,i,w]-u[i]+I[d,k,w] - M*(1-s[d,i,k]) for i in J for d in D for k in K for w in scenarios), name = "waitingtime-4")

        for i in J:
            for k in K:
                for w in scenarios:
                    for j in J:
                        model.addGenConstrIndicator(s[i,j,k],True,(W[j,k,w]==W[i,k,w]+I[i,k,w]+para.travelT[i,j,w]+para.svcT[i,w]-u[j]+u[i]), name = "waitingtime-1")
                    for d in D:
                        model.addGenConstrIndicator(s[i,d,k],True,(O[k,w]==W[i,k,w]+I[i,k,w]+para.travelT[i,d,w]+para.svcT[i,w]-para.totOprT[k]+u[i]), name = "overtime")
                        model.addGenConstrIndicator(s[d,i,k],True,(W[i,k,w]==I[d,k,w]+para.travelT[d,i,w]-u[i]), name = "waitingtime-2")

    # Optimize model
        model.optimize()

    ### Print results
    # Assignments
        for k in vehicles:
            if x[k.name].X < 0.5:
                vehStr = "Vehicle {} is not used".format(k.name)
            else:
                for d in D:
                    if z[k.name,d].X > 0.5:
                        k.depot = d
                        vehStr = "Vehicle {} assigned to depot {}.".format(k.name,d)
            print(vehStr, file=f)
    # Routing
        print("",file=f)
        for k in vehicles:
            if x[k.name].X > 0.5:
                cur = k.depot
                route = k.depot
                while True:
                    for i in patients:
                        if s[cur,i.name,k.name].x > 0.5:
                            route += " -> {} (dist={:.2f}, t={:.2f}, proc={:.2f})".format(i.name,para.dist[cur,i.name],u[i.name].x,i.dur)
                            cur = i.name
                    for j in D:
                        if s[cur,j,k.name].x > 0.5:
                            route += " -> {} (dist={:.2f})".format(j, para.dist[cur,j])
                            cur = j
                            break
                    if cur == k.depot:
                        break
                print("Vehicle {}'s route: {}".format(k.name, route),file=f)

        outputU = {i:u[i].x for i in J}
        outputS = {(i,j,k):s[i,j,k].x for i in N for j in N for k in K}
        outputX = {k:x[k].x for k in K}
        return outputU, outputS, outputX

    except gp.GurobiError as e:
        print ('Error code' + str (e. errno ) + ': ' + str(e))
        pass
    except AttributeError :
        print ('Encountered an attribute error')
        pass

def SecondStage(para,u,s,x,w):
    try:
    # Create a new model
        model = gp.Model ("SSMILP")

    # Create variables
        O = model.addVars(K, lb = 0.0, name = "O")                       # overtime of veh k
        I = model.addVars(N, K, lb = 0.0, name = "I")                    # idletime at patient i of veh k
        W = model.addVars(N, K, lb = 0.0, name = "W")                    # waitingtime at patient i of veh k

    # Set objective
        objective = gp.quicksum(para.oprCost[i]*x[i] for i in K) \
            + gp.quicksum(para.traCost[i,j]*para.travelT[i,j,w]*s[i,j,k] for i in N for j in N for k in K) \
                + gp.quicksum(para.waitingPenalty*W[i,k] + para.idlePenalty*I[i,k] + para.overtimePenalty*O[k] for k in K for i in N)
        model.setObjective(objective, GRB.MINIMIZE)

    # Add constraints
        M = 60*60
        model.addConstrs((W[j,k]==para.travelT[i,j,w]+para.svcT[i,w]-u[j]+u[i]+W[i,k]+I[i,k] for i in J for j in J for k in K if s[i,j,k]==1), name = "waitingtime-1")
        model.addConstrs((O[k]==para.travelT[i,d,w]+para.svcT[i,w]-para.totOprT[k]+u[i]+W[i,k]+I[i,k] for i in J for d in D for k in K if s[i,d,k]==1), name = "overtime")
        model.addConstrs((W[i,k]==para.travelT[d,i,w]-u[i]+I[d,k] for i in J for d in D for k in K if s[d,i,k]==1), name = "waitingtime-2")

    # Optimize model
        model.optimize()

    # print optimal solutions
        for k in vehicles:
            if x[k.name] > 0.5:
                vehStr = "\tOvertime: {}".format(O[k.name].x)
                for i in N:
                    vehStr += "\n\tPatient {}: Idletime = {}, Waitingtime = {}".format(i,I[i,k.name].x,W[i,k.name].x)
                print("Vehicle {}:\n {}".format(k.name, vehStr),file=f)

        return model.objVal

    except gp.GurobiError as e:
        print ('Error code' + str (e. errno ) + ': ' + str(e))
        pass
    except AttributeError :
        print ('Encountered an attribute error')
        pass

def printScen(scenStr):
    sLen = len(scenStr)
    print("\n" + "*"*sLen + "\n" + scenStr + "\n" + "*"*sLen + "\n", file=f)

### Helper Classes
class Vehicle():
    def __init__(self, name, cap, totOprTime, oprCost):
        self.name = name
        self.cap = cap
        self.totOprTime = totOprTime
        self.oprCost = oprCost

    def __str__(self):
        return f"Vehicle: {self.name}\n  Capacity: {self.cap}\n  Total Operation Time: {self.totOprTime}\n \
            Operation Cost: {self.oprCost}"

class Demand():
    def __init__(self, name, capacity, duration):
        self.name = name
        self.capacity = capacity
        self.dur = duration

    def __str__(self):
        about = f"Demand: {self.name}\n  Capacity: {self.capacity}\n  Duration: {self.dur}"
        return about

class Patient():
    def __init__(self, name, locX, locY, tStart, tEnd, demandType, demand, dur):
        self.name = name
        self.loc = [locX,locY]
        self.dType = demandType
        self.tStart = tStart
        self.tEnd = tEnd
        self.demand = demand
        self.dur = dur

    def __str__(self):
        return f"Patient: {self.name}\n  Location: {self.locX, self.locY}\n  Demand: {self.dType.name}\n  Capacity: {self.dType.capacity}\n  Duration: {self.dur}\n  Start time: {self.tStart}\n  End time: {self.tEnd}"

class Depot():
    def __init__(self, name, locX, locY):
        self.name = name
        self.loc = [locX,locY]

    def __str__(self):
        return f"Depot: {self.name}\n Location: {self.loc}"

class Node():
    def __init__(self, name, type, loc):
        self.name = name
        self.type = type
        self.loc = loc

    def __str__(self):
        return f"Node: {self.name}\n Type: {self.type}\n Location: {self.loc}"


# Read data
def LoadData(file):
    filename = file
    ws = pd.read_excel(filename, sheet_name='Patients')
    patients = []
    for i, row in ws.iterrows():
        this = Patient(*row)
        patients.append(this)

    ws = pd.read_excel(filename, sheet_name='Depots')
    depots = []
    for i,row in ws.iterrows():
        this = Depot(*row)
        depots.append(this)

    ws = pd.read_excel(filename, sheet_name='Vehicles')
    vehicles = []
    for i,row in ws.iterrows():
        this = Vehicle(*row)
        vehicles.append(this)

    nodes = []
    for row in patients:
        this = Node(row.name, 'Patient', row.loc)
        nodes.append(this)
    for row in depots:
        this = Node(row.name, 'Depot', row.loc)
        nodes.append(this)
    
    return vehicles, patients, depots, nodes


class Parameters:
    def __init__(self, vehicles, patients, depots, nodes):
        self.cap = {k.name: k.cap for k in vehicles}             # capacity
        self.totOprT = {k.name: k.totOprTime for k in vehicles}  # total operation time
        self.oprCost = {k.name: k.oprCost for k in vehicles}     # operation cost
        self.demand = {i.name: i.demand for i in patients}
        self.tStart = {i.name: i.tStart for i in patients}
        self.tEnd = {i.name: i.tEnd for i in patients}
        self.dist = {(l.name,l.name):0 for l in nodes}                     # distance between 2 nodes
        for i, l1 in enumerate(nodes):
            for j, l2 in enumerate(nodes):
                if i < j:
                    self.dist[l1.name,l2.name] = haversine(*l1.loc,*l2.loc)
                    self.dist[l2.name,l1.name] = self.dist[l1.name,l2.name]
        self.traCost = {(i.name, j.name):1 for i in nodes for j in nodes}      # travel cost
        sig = 0.2
        self.travelT = {(i,j,w):self.dist[i,j] for i,j in self.dist for w in scenarios}
        self.svcT = {(i.name,w): i.dur for i in patients for w in scenarios}
        for w in scenarios:
            # sampling travel times for each arc
            samp = range(3)
            case = np.random.choice(samp,size=1,p=[0.8,0.15,0.05])
            if case == 0:
                speed = np.random.normal(60,60*sig)
            elif case == 1:
                speed = np.random.normal(50,50*sig)
            else:
                speed = np.random.normal(30,30*sig)
            for i,j in self.dist:
                self.travelT[i,j] = self.dist[i,j]*60/speed

            # sampling service times for each patient
            samp = range(4)
            for i in patients:
                case = np.random.choice(samp,size=1,p=[0.3,0.05,0.5,0.15])
                if case == 0:
                    self.svcT[i.name,w] = np.random.exponential(45)
                elif case == 1:
                    self.svcT[i.name,w] = np.random.exponential(60)
                elif case == 2:
                    self.svcT[i.name,w] = np.random.exponential(30)
                else:
                    self.svcT[i.name,w] = np.random.exponential(90)

        self.traTMean = {(i,j):self.dist[i,j]*60/(60*0.8+50*0.15+30*0.05) for i,j in self.dist} 

        for i in patients:
            temp = 0
            for w in scenarios:
                temp += self.svcT[i.name,w]
            i.dur = temp/len(scenarios)

        self.svcTMean = {i.name: i.dur for i in patients}

        self.waitingPenalty = 1
        self.idlePenalty = 2
        self.overtimePenalty = 10
        pass

class Sets:
    def __init__(self,name,numscenario = 5):
        file = "./data{}.xlsx".format(name)
        self.vehicle, self.patient, self.depot, self.node = LoadData(file)
        # sets
        self.K = [k.name for k in self.vehicle]
        self.J = [j.name for j in self.patient]
        self.D = [d.name for d in self.depot]
        self.N = [n.name for n in self.node]
        # parameters
        self.scenario = range(numscenario)                                # scenarios
        self.p = [1/len(self.scenario) for i in self.scenario]           # probability
        self.f = open("output{}.txt".format(name), "w")


sets = Sets("(4-10)",10)

K = sets.K
J = sets.J
D = sets.D
N = sets.N
scenarios = sets.scenario
p = sets.p
f = sets.f
vehicles = sets.vehicle
patients = sets.patient
para = Parameters(sets.vehicle, sets.patient, sets.depot, sets.node)
printScen("Solving TSMILP Model using Benders Decomposition")
start_time = time.time()
# u,s,x = TSMILP()
# printScen("Solving Second-Stage Model when scenario is realized")
# SecondStage(u,s,x,1)
m = Benders.Benders(para,sets)
m.optimize()
end_time = time.time()
printScen("time taken = "+str(end_time-start_time))


