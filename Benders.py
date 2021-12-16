import gurobipy as gp
from gurobipy import GRB

####
#   Benders decomposition via Gurobi + Python
####

class expando(object):
    pass
class Subproblem:
    def __init__(self, MP, scenario=0):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.data.MP = MP
        self.scenario = scenario
        self.update_fixed_vars(MP)
        self._build_model()

    def optimize(self):
        self.model.optimize()

    def _build_model(self):
        self.model = gp.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model

        self.variables.O = m.addVars(K, lb = 0.0, name = "O")                          # overtime of veh k
        self.variables.I = m.addVars(N, K, lb = 0.0, name = "I")                       # idletime at patient i of veh k
        self.variables.W = m.addVars(N, K, lb = 0.0, name = "W")                       # waitingtime at patient i of veh k
        self.variables.fix_u = m.addVars(J, lb = 0.0, name = "fix_u")                          
        self.variables.fix_s = m.addVars(N, N, K, lb = 0.0, name = "fix_s")

        m.update()

    def _build_objective(self):
        m = self.model
        w = self.scenario
        data = self.data.MP.data.para
        objective = gp.quicksum(data.traCost[i,j]*data.travelT[i,j,w]*self.variables.fix_s[i,j,k] for i in N for j in N for k in K) \
                + gp.quicksum(data.waitingPenalty*self.variables.W[i,k] + data.idlePenalty*self.variables.I[i,k] + data.overtimePenalty*self.variables.O[k] for k in K for i in N)
        m.setObjective(
            objective,
            GRB.MINIMIZE)

    def _build_constraints(self):
        m = self.model
        data = self.data.MP.data.para
        
        O = self.variables.O
        I = self.variables.I
        W = self.variables.W
        fix_u = self.variables.fix_u
        fix_s = self.variables.fix_s
        w = self.scenario

        self.constraints.waitingtime_1 = {}
        self.constraints.waitingtime_2 = {}
        self.constraints.overtime1 = {}
        self.constraints.waitingtime_3 = {}
        self.constraints.waitingtime_4 = {}
        self.constraints.overtime2 = {}
        self.constraints.fix_u = {}
        self.constraints.fix_s = {}
        M = 60*60
        for i in J:
            for k in K:
                for j in J:
                    self.constraints.waitingtime_1[i,j,k] = m.addConstr(M*(1-fix_s[i,j,k])+W[j,k]>=W[i,k]+I[i,k]+data.travelT[i,j,w]+data.svcT[i,w]-fix_u[j]+fix_u[i])
                    self.constraints.waitingtime_2[i,j,k] = m.addConstr(-M*(1-fix_s[i,j,k])+W[j,k]<=W[i,k]+I[i,k]+data.travelT[i,j,w]+data.svcT[i,w]-fix_u[j]+fix_u[i])
                for d in D:
                    self.constraints.overtime1[i,d,k] = m.addConstr(M*(1-fix_s[i,d,k])+O[k]>=W[i,k]+I[i,k]+data.travelT[i,d,w]+data.svcT[i,w]-data.totOprT[k]+fix_u[i])
                    self.constraints.overtime2[i,d,k] = m.addConstr(-M*(1-fix_s[i,d,k])+O[k]<=W[i,k]+I[i,k]+data.travelT[i,d,w]+data.svcT[i,w]-data.totOprT[k]+fix_u[i])
                    self.constraints.waitingtime_3[i,d,k] = m.addConstr(M*(1-fix_s[d,i,k])+W[i,k]>=I[d,k]+data.travelT[d,i]-fix_u[i])
                    self.constraints.waitingtime_4[i,d,k] = m.addConstr(-M*(1-fix_s[d,i,k])+W[i,k]<=I[d,k]+data.travelT[d,i]-fix_u[i])
            self.constraints.fix_u[i] = m.addConstr((fix_u[i] == self.data.MP.variables.u[i].x), name = "fix_u")
        for i in N:
            for j in N:
                for k in K:
                    self.constraints.fix_s[i,j,k] = m.addConstr((fix_s[i,j,k] == self.data.MP.variables.s[i,j,k].x), name = "fix_s")

    def update_fixed_vars(self, MP):
        pass

class MasterProblem:
    def __init__(self, para, sets, max_iters=500, verbose=True, numscenarios=5, epsilon=0.001, delta=0.001):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.params = expando()

        self.params.max_iters = max_iters
        self.params.verbose = verbose
        self.params.numscenarios = numscenarios

        global K
        global J
        global D
        global N
        global scenarios
        global p
        global f
        global vehicles
        global patients

        K = sets.K
        J = sets.J
        D = sets.D
        N = sets.N
        scenarios = sets.scenario
        p = sets.p
        f = sets.f
        vehicles = sets.vehicle
        patients = sets.patient

        self._init_benders_params(epsilon=epsilon, delta=delta)
        self._load_data(para)
        self._build_model()

    def _init_benders_params(self, epsilon=0.001, delta=0.001):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.mipgap = []
        self.data.solvetime = []
        self.data.alphas = []
        self.data.lambdas = {}
        self.data.epsilon = epsilon
        self.data.delta = delta
        self.data.ub = GRB.INFINITY
        self.data.lb = -GRB.INFINITY

    def _load_data(self, para):
        self.data.para = para

    def _build_model(self):
        self.model = gp.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        mp = self.model
        self.variables.x ={}
        self.variables.y = {}
        self.variables.z = {}
        self.variables.s = {}
        self.variables.u = {}
        for k in K:
            self.variables.x[k] = mp.addVar(vtype = GRB.BINARY, name = "x")    
            for j in J:            
                self.variables.y[j,k] = mp.addVar(vtype = GRB.BINARY, name = "y")
            for d in D:
                self.variables.z[k,d] = mp.addVar(vtype = GRB.BINARY, name = "z")
            for j in N:
                for i in N:
                    self.variables.s[j,i,k] = mp.addVar(vtype = GRB.BINARY, name = "s")
        for j in J:
            self.variables.u[j] = mp.addVar(lb = 0.0, name = "u")
        self.variables.alpha = mp.addVar(lb=0.0, name = "alpha")
        mp.update()

    def _build_objective(self):
        m = self.model
        objective = gp.quicksum(self.data.para.oprCost[i]*self.variables.x[i] for i in K) + self.variables.alpha
        m.setObjective(objective, GRB.MINIMIZE)

    def _build_constraints(self):
        m = self.model

        m.addConstrs((gp.quicksum(self.variables.z[k,d] for d in D)==1 for k in K), name = "assign-1")
        m.addConstrs((gp.quicksum(self.variables.y[i,k] for k in K)==1 for i in J), name = "assign-2")
        m.addConstrs((gp.quicksum(self.variables.s[i,j,k] for j in N) == self.variables.y[i,k] for k in K for i in J), name="tour1")
        m.addConstrs((gp.quicksum(self.variables.s[i,j,k] for i in N) == self.variables.y[j,k] for k in K for j in J), name="tour2")
        m.addConstrs((gp.quicksum(self.variables.s[d,i,k] for i in J)==self.variables.z[k,d]*self.variables.x[k] for k in K for d in D), name = "samedepot1")
        m.addConstrs((gp.quicksum(self.variables.s[i,d,k] for i in J)==self.variables.z[k,d]*self.variables.x[k] for k in K for d in D), name = "samedepot2")
        m.addConstrs((gp.quicksum(self.variables.y[i,k]*self.data.para.demand[i] for i in J)<=self.data.para.cap[k]*self.variables.x[k] for k in K), name = "capacity")
        m.addConstrs((self.data.para.tStart[i]<=self.variables.u[i] for i in J), name = "timewindow-1")
        m.addConstrs((self.variables.u[i]<=self.data.para.tEnd[i] for i in J), name = "timewindow-2")
        M = {(i,j) : 24*60 + self.data.para.svcTMean[i] + self.data.para.traTMean[i,j] for i in J for j in J}
        m.addConstrs((self.variables.u[j]>=self.variables.u[i]+self.data.para.traTMean[i,j]+self.data.para.svcTMean[i]-M[i,j]*(1-gp.quicksum(self.variables.s[i,j,k] for k in K)) for i in J for j in J), name = "planned_arri_time")
        
        self.constraints.cuts = {}

    def _update_bounds(self):
        z_sub = sum(p[w]*self.submodels[w].model.ObjVal for w in scenarios)
        z_master = self.model.ObjVal
        self.data.ub = z_master - self.variables.alpha.x + z_sub
        try:
            self.data.lb = self.model.ObjBound
        except gp.GurobiError:
            self.data.lb = self.model.ObjVal
        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)
        self.data.mipgap.append(self.model.params.IntFeasTol)
        self.data.solvetime.append(self.model.Runtime)
        self.data.alphas.append(self.variables.alpha.x)

    def optimize(self):
        self.model.optimize()

        self.submodels = {w:Subproblem(self,scenario= w) for w in scenarios}
        [sm.optimize() for sm in self.submodels.values()]

        self._update_bounds()

        while((self.data.ub > self.data.lb + self.data.delta 
        or self.data.ub - self.data.lb > abs(self.data.epsilon * self.data.lb)) and len(self.data.cutlist) < self.params.max_iters):
        
            if self.params.verbose:
                print('********')
                print('* Benders\' step {0}:'.format(len(self.data.upper_bounds)))
                print('* Upper bound: {0}'.format(self.data.ub))
                print('* Lower bound: {0}'.format(self.data.lb))
                print('********')
            self._do_benders_step()
        pass

        self._printResult()
    
    def _do_benders_step(self):
        self._add_cut()
        self.model.optimize()
        [sm.update_fixed_vars(self) for sm in self.submodels.values()]
        [sm.optimize() for sm in self.submodels.values()]
        self._update_bounds()

    def _add_cut(self):
        cut = len(self.data.cutlist)
        self.data.cutlist.append(cut)

        sens_s = {
            (i,j,k): sum(p[w] * self.submodels[w].constraints.fix_s[i,j,k].pi for w in scenarios)
             for i in N for j in N for k in K}
        self.data.lambdas[cut] = sens_s
        sens_u = {j:sum(p[w] * self.submodels[w].constraints.fix_u[j].pi for w in scenarios) for j in J}
        # Get subproblem objectives
        z_sub = sum(p[w] * self.submodels[w].model.ObjVal for w in scenarios)
        # Generate cut
        self.constraints.cuts[cut] = self.model.addConstr(
            self.variables.alpha,
            GRB.GREATER_EQUAL,
            z_sub +
            gp.quicksum(sens_s[i,j,k] * self.variables.s[i,j,k] for i in N for j in N for k in K) -
            sum(sens_s[i,j,k] * self.variables.s[i,j,k].x for i in N for j in N for k in K) +
            gp.quicksum(sens_u[j] * (self.variables.u[j] - self.variables.u[j].x) for j in J)
        )

    def _printResult(self):
    # Assignments
        for k in vehicles:
            if self.variables.x[k.name].X < 0.5:
                vehStr = "Vehicle {} is not used".format(k.name)
            else:
                for d in D:
                    if self.variables.z[k.name,d].X > 0.5:
                        k.depot = d
                        vehStr = "Vehicle {} assigned to depot {}.".format(k.name,d)
            print(vehStr, file=f)
    # Routing
        print("",file=f)
        for k in vehicles:
            if self.variables.x[k.name].X > 0.5:
                cur = k.depot
                route = k.depot
                while True:
                    for i in patients:
                        if self.variables.s[cur,i.name,k.name].x > 0.5:
                            route += " -> {} (dist={:.2f}, t={:.2f}, proc={:.2f})".format(i.name,self.data.para.dist[cur,i.name],self.variables.u[i.name].x,i.dur)
                            cur = i.name
                    for j in D:
                        if self.variables.s[cur,j,k.name].x > 0.5:
                            route += " -> {} (dist={:.2f})".format(j, self.data.para.dist[cur,j])
                            cur = j
                            break
                    if cur == k.depot:
                        break
                print("Vehicle {}'s route: {}".format(k.name, route),file=f)
        # Waiting Idle Overtime
        print("",file=f)
        O = {}
        I = {}
        W = {}
        ot = []
        it = []
        wt = []
        for k in K:
            O[k] = sum(self.submodels[w].variables.O[k].x for w in scenarios)/len(scenarios)
            for i in N:
                I[i,k] = sum(self.submodels[w].variables.I[i,k].x for w in scenarios)/len(scenarios)
                W[i,k] = sum(self.submodels[w].variables.W[i,k].x for w in scenarios)/len(scenarios)
        for k in K:
            if self.variables.x[k].x > 0.5:
                vehStr = "\tOvertime: {}".format(O[k])
                ot.append(O[k])
                for i in J:
                    if self.variables.y[i,k].x > 0.5:
                        vehStr += "\n\tPatient {}: Idletime = {}, Waitingtime = {}".format(i,I[i,k],W[i,k])
                        it.append(I[i,k])
                        wt.append(W[i,k])
                for d in D:
                    if self.variables.z[k,d].x > 0.5:
                        vehStr += "\n\tDepot {}: Idletime = {}, Waitingtime = {}".format(d,I[d,k],W[d,k])
                        it.append(I[d,k])
                        wt.append(W[d,k])
                print("Vehicle {}:\n {}".format(k, vehStr),file=f)
        # Optimality Gap
        print("",file=f)
        print("Benders Iterations: {}".format(len(self.data.cutlist)), file=f)
        print("Objective Value: {:.4f}".format(self.model.ObjVal), file=f)
        print("Average Idle Time: {:.2f}".format(sum(it)/len(it)),file=f)
        print("Average Waiting Time: {:.2f}".format(sum(wt)/len(wt)),file=f)
        print("Average Over Time: {:.2f}".format(sum(ot)/len(ot)),file=f)
        print("Optimality Gap: {:.2%}".format((self.data.ub-self.data.lb)/self.data.lb), file=f)