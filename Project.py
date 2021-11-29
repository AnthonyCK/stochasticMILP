import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

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
        self.loc = np.array([locX,locY])
        self.dType = demandType
        self.tStart = tStart
        self.tEnd = tEnd
        self.demand = demand
        self.dur = dur

    def __str__(self):
        return f"Patient: {self.name}\n  Location: {self.locX, self.locY}\n  Demand: {self.dType.name}\n  Capacity: {self.dType.capacity}\n  Duration: {self.dur}\n  Start time: {self.tStart}\n  End time: {self.tEnd}"

class Depot():
    def __init__(self, name, locX, locY):
        self.name = str(name) + '_D'
        self.loc = np.array([locX,locY])

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
filename = './patients.xlsx'
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


w = 4   # of scenarios
scenarios = range(w)
p = [1/w for i in scenarios]        # probability

# sets
K = [k.name for k in vehicles]
J = [j.name for j in patients]
D = [d.name for d in depots]
N = [n.name for n in nodes]

# parameters
cap = {k.name: k.cap for k in vehicles}             # capacity
totOprT = {k.name: k.totOprTime for k in vehicles}  # total operation time
oprCost = {k.name: k.oprCost for k in vehicles}     # operation cost
demand = {i.name: i.demand for i in patients}
tStart = {i.name: i.tStart for i in patients}
tEnd = {i.name: i.tEnd for i in patients}

dist = {(l.name,l.name):0 for l in nodes}                     # distance between 2 nodes
for i, l1 in enumerate(nodes):
    for j, l2 in enumerate(nodes):
        if i < j:
            dist[l1.name,l2.name] = np.linalg.norm(l1.loc-l2.loc)
            dist[l2.name,l1.name] = dist[l1.name,l2.name]
traCost = {(i.name, j.name):1 for i in nodes for j in nodes}      # travel cost

    # travel time
travelT = {(i,j,w):dist[i,j]*60/50 for i,j in dist for w in scenarios}
for w in scenarios:
    speed = np.random.normal(50)
    for i,j in dist:
        travelT[i,j] = dist[i,j]*60/speed
traTMean = {(i,j):dist[i,j]*60/50 for i,j in dist} 
  # service time
svcT = {(i.name,w): i.dur for i in patients for w in scenarios}
svcTMean = {i.name: i.dur for i in patients}

waitingPenalty = 2
idlePenalty = 1
overtimePenalty = 10

def FirstStage():
    try:
    # Create a new model
        model = gp.Model ("FSMILP")

    # Create variables
        x = model.addVars(K, vtype = GRB.BINARY, name = "x")                # operate veh k
        y = model.addVars(J, K, vtype = GRB.BINARY, name = "y")             # assign patient i to veh k
        z = model.addVars(K, D, vtype = GRB.BINARY, name = "z")             # assign veh k to depot d
        u = model.addVars(J, lb = 0.0, name = "u")                          # planned arrival time at i
        s = model.addVars(N, N, K, vtype = GRB.BINARY, name = "s")          # assign arc (i,j) to veh k

        O = model.addVars(K, lb = 0.0, name = "O")                          # overtime of veh k
        I = model.addVars(N, K, lb = 0.0, name = "I")                       # idletime at patient i of veh k
        W = model.addVars(N, K, lb = 0.0, name = "W")                       # waitingtime at patient i of veh k
        model.update()
    # Set objective
        objective = gp.quicksum(oprCost[i]*x[i] for i in K)\
             + gp.quicksum(p[w]*(gp.quicksum(traCost[i,j]*travelT[i,j,w]*s[i,j,k] for i in N for j in N for k in K) + gp.quicksum(waitingPenalty*W[i,k] + idlePenalty*I[i,k] + overtimePenalty*O[k] for k in K for i in N)) for w in scenarios)
        model.setObjective(objective, GRB.MINIMIZE)

    # Add constraints
        model.addConstrs((gp.quicksum(z[k,d] for d in D)==1 for k in K), name = "assign-1")
        model.addConstrs((gp.quicksum(y[i,k] for k in K)==1 for i in J), name = "assign-2")
        model.addConstrs((y[i,k] <=x[k] for i in J for k in K), name = "assign-3")
        model.addConstrs((gp.quicksum(s[i,j,k] for j in N) == y[i,k] for k in K for i in J), name="tour1")
        model.addConstrs((gp.quicksum(s[i,j,k] for i in N) == y[j,k] for k in K for j in J), name="tour2")
        model.addConstrs((gp.quicksum(s[d,i,k] for i in J)==z[k,d]*x[k] for k in K for d in D), name = "samedepot1")
        model.addConstrs((gp.quicksum(s[i,d,k] for i in J)==z[k,d]*x[k] for k in K for d in D), name = "samedepot2")
        model.addConstrs((gp.quicksum(s[i,j,k]*demand[i] for i in J for j in J if j != i)<=cap[k]*x[k] for k in K), name = "capacity")
        model.addConstrs((tStart[i]<=u[i] for i in J), name = "timewindow-1")
        model.addConstrs((u[i]<=tEnd[i] for i in J), name = "timewindow-2")
        M = {(i,j) : 24*60 + svcTMean[i] + traTMean[i,j] for i in J for j in J}
        model.addConstrs((u[j]>=u[i]+traTMean[i,j]+svcTMean[i]-M[i,j]*(1-gp.quicksum(s[i,j,k] for k in K)) for i in J for j in J), name = "planned_arri_time")

        M = 60*60
        model.addConstrs((W[j,k]>=(travelT[i,j,w]+svcT[i,w]+W[i,k]+I[i,k])-M*(1-s[i,j,k])-u[j]+u[i] for i in J for j in J for k in K for w in scenarios), name = "waitingtime-1")
        model.addConstrs((O[k]>=(travelT[i,d,w]+svcT[i,w]+W[i,k]+I[i,k])-M*(1-s[i,d,k])-totOprT[k]+u[i] for i in J for d in D for k in K for w in scenarios), name = "overtime")
        model.addConstrs((W[i,k]>=(travelT[d,i,w]+I[d,k])-M*(1-s[d,i,k])-u[i] for i in J for d in D for k in K for w in scenarios), name = "waitingtime-2")

    # Optimize model
        model.optimize()
        # model.computeIIS()
        # model.write("model.lp")
        # if model.status == GRB.INFEASIBLE:
        #     vars = model.getVars()
        #     ubpen = [1.0]*model.numVars
        #     model.feasRelax(1, False, vars, None, ubpen, None, None)
        #     model.optimize()

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
                            route += " -> {} (dist={:.2f}, t={:.2f}, proc={})".format(i.name,dist[cur,i.name],u[i.name].x,i.dur)
                            cur = i.name
                    for j in D:
                        if s[cur,j,k.name].x > 0.5:
                            route += " -> {} (dist={:.2f})".format(j, dist[cur,j])
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

def SecondStage(u,s,x,w):
    try:
    # Create a new model
        model = gp.Model ("SSMILP")

    # Create variables
        O = model.addVars(K, lb = 0.0, name = "O")                       # overtime of veh k
        I = model.addVars(N, K, lb = 0.0, name = "I")                    # idletime at patient i of veh k
        W = model.addVars(N, K, lb = 0.0, name = "W")                    # waitingtime at patient i of veh k

    # Set objective
        objective = gp.quicksum(traCost[i,j]*travelT[i,j,w]*s[i,j,k] for i in N for j in N for k in K) + gp.quicksum(waitingPenalty*W[i,k] + idlePenalty*I[i,k] + overtimePenalty*O[k] for k in K for i in N)
        model.setObjective(objective, GRB.MINIMIZE)

    # Add constraints
        M = 60*60
        model.addConstrs((W[j,k]>=(travelT[i,j,w]+svcT[i,w]+W[i,k]+I[i,k])-M*(1-s[i,j,k])-u[j]+u[i] for i in J for j in J for k in K), name = "waitingtime-1")
        model.addConstrs((O[k]>=(travelT[i,d,w]+svcT[i,w]+W[i,k]+I[i,k])-M*(1-s[i,d,k])-totOprT[k]+u[i] for i in J for d in D for k in K), name = "overtime")
        model.addConstrs((W[i,k]>=(travelT[d,i,w]+I[d,k])-M*(1-s[d,i,k])-u[i] for i in J for d in D for k in K), name = "waitingtime-2")

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

# Test Model
f = open("output.txt", "w")
printScen("Solving First-Stage Model")
u,s,x = FirstStage()
printScen("Solving Second-Stage Model")
SecondStage(u,s,x,1)