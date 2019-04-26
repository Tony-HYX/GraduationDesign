import sys
import time
import copy
import math
sys.path.insert(0, '../lib/')

import LogicLayer as LL
from zoopt import Dimension, Objective, Parameter, Opt, Solution

'''
# logiclayer feature
class LL_feature:
    mapping = {}
    rules = []
    def __init__(self, m, r):
        self.mapping = m
        self.rules = r

    # evaluate ONE example
    def eval(self, ex):
        # apply mapping
        ex_symbs_n = apply_mapping(ex, self.mapping)
        (ex_symbs, n_pos) = remove_nulls(ex_symbs_n)
        ex_term = LL.PlTerm(ex_symbs) # LL term
        return LL.evalInstFeature(ex_term, LL.PlTerm(self.rules))
'''

# class of consistent result score
class consist_result:
    score = 0
    consistent_ex_ids = []
    abduced_exs = [] # 
    abduced_map = {} # mapping of NN output to symbols
    abduced_rules = [] # Prolog rules
    def __init__(self):
        pass

    def __init__(self, max_score, max_exs, max_map, indices):
        self.score = max_score
        self.abduced_exs = max_exs
        self.abduced_map = max_map
        #self.abduced_rules = max_rules
        self.consistent_ex_ids = indices
    def copy(self):
        return consist_result(self.score,
                              copy.deepcopy(self.abduced_exs),#self.abduced_exs.copy(),
                              self.abduced_map.copy(),
                              #self.abduced_rules.copy(),
                              self.consistent_ex_ids.copy())
    '''
    def to_feature(self):
        feat = LL.conInstsFeature(self.abduced_rules)
        return LL_feature(self.abduced_map, feat)
    '''

def gen_mappings(chars, symbs):
    n_char = len(chars);
    n_symbs = len(symbs);
    if n_char != n_symbs:
        print('Characters and symbols size dosen\'t match.')
        return
    from itertools import permutations;
    mappings = []; # returned mappings
    perms = permutations(symbs)
    for p in perms:
        mappings.append(dict(zip(chars, list(p))))
    return mappings
    
'''
def apply_mapping(chars, mapping):
    re = []
    for c in chars:
        if c == '_': # leave vars unchanged
            re.append(c)
        elif not (c in mapping):
            print('Wrong character for mapping.')
            return
        else:
            re.append(mapping[c])
    return re
'''

def apply_mapping_chess(ex, mapping):
    re = copy.deepcopy(ex)#ex.copy()
    for i in range(len(re[0])):
        if type(re[0][i][2]) == str: # leave vars unchanged
            continue
        elif not (re[0][i][2] in mapping):
            print('Wrong character for mapping.')
            return
        else:
            re[0][i][2] = mapping[re[0][i][2]]
    return re

'''
def flatten(l):
    return [item for sublist in l for item in sublist]

# reformulate identifiers from flat list to list of lists as examples
def reform_ids(exs, var_ids):
    exs_f = flatten(exs)
    assert len(exs_f) == len(var_ids)
    re = []
    i = 0
    for e in exs:
        j = 0
        ids = []
        while j < len(e):
            ids.append(var_ids[i + j])
            j += 1
        re.append(ids)
        i = i + j
    return re
'''

# reformulate identifiers from flat list to list of lists as examples
def reform_ids_chess(exs, var_ids):
    assert num_of_chess(exs) == len(var_ids)
    re = []
    i = 0
    for e in exs:
        j = 0
        ids = []
        while j < len(e[0]):
            ids.append(var_ids[i + j])
            j += 1
        re.append(ids)
        i = i + j
    return re


# substitute const in examples to vars according to the var identifiers (flatten)
def sub_vars_chess(exs, var_ids):
    var_cnt = 1
    subd_exs = copy.deepcopy(exs)
    for i in range(0, len(subd_exs)):
        ex = subd_exs[i][0];
        var_id = var_ids[i];
        assert len(ex) == len(var_id)
        for j in range(0, len(ex)):
            if var_id[j]:
                ex[j][2] = 'X'+str(var_cnt)# replace a variable
                subd_exs[i][1].append('X'+str(var_cnt))
                var_cnt += 1
    return subd_exs
    
'''
# return a consistent score (number maximum consistent examples) of given list of examples and variable indicators
def consistent_score(exs, var_ids, maps):
    # for debug
    #print('vars: ', end = '\t')
    #print(var_ids)
    
    max_score = 0
    max_exs = []
    max_map = {}
    max_rules = []
    max_subd_exs_ids = []
    max_null_pos = []
    
    subd_exs_all = sub_vars(exs, var_ids) # examples been replaced variables
    subd_exs_ids = []
    #count = 0
    t = time.time()
    for i in range(0, len(subd_exs_all)):
        subd_exs_ids.append(i)

        subd_exs = []
        for j in subd_exs_ids:
            subd_exs.append(subd_exs_all[j])

        got_con_sol = False
        # do mapping and evaluation, TODO: possible for multithread
        for m in maps:
            #LL.gc()
            #LL.trimStacks() # IMPORTANT!!
            mapped_subd_exs = [] # list of plterms
            
            null_pos = []
            for e in subd_exs:
                e_symbs_n = apply_mapping(e, m)
                (e_symbs, n_pos) = remove_nulls(e_symbs_n)
                null_pos.append(n_pos)
                mapped_subd_exs.append(LL.PlTerm(e_symbs))
            mapped_subd_term = LL.PlTerm(mapped_subd_exs)
            con_sol = LL.abduceConInsts(mapped_subd_term) # consistent solution

            if con_sol:
                got_con_sol = True
            if con_sol and max_score < len(subd_exs):
                max_rules = copy.deepcopy(con_sol.py())
                max_score = len(subd_exs)
                max_subd_exs_ids = subd_exs_ids.copy()
                max_map = m.copy()
                max_null_pos = null_pos.copy()
                
                if max_score == len(mapped_subd_exs):
                    break
        if not got_con_sol:
            subd_exs_ids.pop()

    abduced_exs = exs.copy()
    inv_m = { v: k for k, v in max_map.items() } # inverted map
    #print(max_subd_exs_ids)
    for j in range(0, len(max_subd_exs_ids)):
        # add nulls back
        ex_n = add_nulls(max_rules[j].copy(), max_null_pos[j])
        mapped_ex = apply_mapping(ex_n, inv_m)
        abduced_exs[max_subd_exs_ids[j]] = mapped_ex
    
    #print(max_score)
    
    re = consist_result(max_score, abduced_exs, max_map,
                        max_rules, max_subd_exs_ids)
    return re

# this score evaluation does not iterate on mappings
def consistent_score_mapped(exs, var_ids, m):
    max_score = 0
    max_exs = []
    max_rules = []
    subd_exs = sub_vars(exs, var_ids) # examples been replaced variables

    mapped_subd_exs = [] # list of plterms
    inv_m = { v: k for k, v in m.items() } # inverted map
    null_pos = []
    for e in subd_exs:
        e_symbs_n = apply_mapping(e, m)
        (e_symbs, n_pos) = remove_nulls(e_symbs_n)
        null_pos.append(n_pos)
        mapped_subd_exs.append(LL.PlTerm(e_symbs))
    mapped_subd_term = LL.PlTerm(mapped_subd_exs)
    con_sol = LL.abduceConInsts(mapped_subd_term) # consistent solution

    if con_sol:
        max_rules = copy.deepcopy(con_sol.py())
        max_subd_exs = con_sol.py().copy()
        max_exs = []
        for k in range(0, len(max_subd_exs)):
            # add nulls back
            max_subd_exs_n = add_nulls(max_subd_exs[k], null_pos[k])
            # map back
            max_exs.append(apply_mapping(max_subd_exs_n, inv_m))
        abduced_exs = exs.copy()
        for i in range(0, len(max_exs)):
            abduced_exs[i] = max_exs[i]
        max_score = len(max_exs)
        re = consist_result(max_score, abduced_exs, m, max_rules, [])
        return re
    else:
        return None
'''

# this score evaluation does not iterate on mappings
def consistent_score_mapped_chess(exs, var_ids, m):
    max_score = 0
    max_exs = []
    max_rules = []
    subd_exs = sub_vars_chess(exs, var_ids) # examples been replaced variables
    mapped_subd_exs = [] # list of plterms
    inv_m = { v: k for k, v in m.items() } # inverted map
    for e in subd_exs:
        e_mapped = apply_mapping_chess(e, m)
        mapped_subd_exs.append(LL.PlTerm(e_mapped))
    mapped_subd_term = LL.PlTerm(mapped_subd_exs)
    con_sol = LL.abduceChessInstFeature(mapped_subd_term) # consistent solution

    if con_sol:
        #max_rules = copy.deepcopy(con_sol.py())
        max_subd_exs = con_sol[0].py() #Only the first solution is used
        for k in range(len(max_subd_exs)):
            # map back
            max_exs.append(apply_mapping_chess(max_subd_exs[k], inv_m))
        max_score = len(max_exs)
        re = consist_result(max_score, max_exs, m, [])
        return re
    else:
        return None
        
'''
# this one does not iterate on mappings and return a set of equation sets
def consistent_score_sets(exs, var_ids_flat, mapping):
    var_ids = reform_ids(exs, var_ids_flat)
    lefted_ids = [i for i in range(0, len(exs))]
    consistent_set_size = []
    consistent_res = []
    # find consistent sets
    while lefted_ids:

        temp_ids = []
        temp_ids.append(lefted_ids.pop(0))
        max_con_ids = []
        max_con_res = None
        found = False
        for i in range(-1, len(exs)):
            if (not i in temp_ids) and (i >= 0):
                temp_ids.append(i)
            # test if consistent
            temp_exs = []
            temp_var_ids = []

            for i in temp_ids:
                temp_exs.append(exs[i])
                temp_var_ids.append(var_ids[i])
            con_res = consistent_score_mapped(temp_exs, temp_var_ids, mapping)
            if not con_res:
                if len(temp_ids) > 1:
                    temp_ids.pop()
            else:
                if len(temp_ids) > len(max_con_ids):
                    found = True
                    max_con_ids = temp_ids.copy()
                    max_con_res = con_res.copy()
                    
                    #print('con:', end = '\t')
                    #print(temp_ids)
                    #print(max_con_res.abduced_rules)
                    #print('left:', end = '\t')
                    #print([i for i in lefted_ids if i not in max_con_ids])
                    

        removed = [i for i in lefted_ids if i in max_con_ids]

        if found:
            #input('Hit any key to continue')
            max_con_res.consistent_ex_ids = max_con_ids.copy()
            consistent_res.append(max_con_res.copy())
            consistent_set_size.append(len(removed) + 1)
            lefted_ids = [i for i in lefted_ids if i not in max_con_ids]

    consistent_set_size.sort()
    score = 0
    for i in range(0, len(consistent_set_size)):
        score += math.exp(-i) * consistent_set_size[i]
    return (score, consistent_res)
'''

# this one does not iterate on mappings and return a set of equation sets
def consistent_score_sets_chess(exs, var_ids_flat, mapping):
    var_ids = reform_ids_chess(exs, var_ids_flat)
    lefted_ids = [i for i in range(len(exs))]
    consistent_set_size = []
    consistent_res = []
    # find consistent sets
    while lefted_ids:
        temp_ids = []
        temp_ids.append(lefted_ids.pop(0)) #Every time force a example that is not used to appear
        max_con_ids = []
        max_con_res = None
        found = False
        for i in range(-1, len(exs)):
            if (not i in temp_ids) and (i >= 0):
                temp_ids.append(i)
            # test if consistent
            temp_exs = []
            temp_var_ids = []

            for i in temp_ids:
                temp_exs.append(exs[i])
                temp_var_ids.append(var_ids[i])

            con_res = consistent_score_mapped_chess(temp_exs, temp_var_ids, mapping)
            if not con_res:
                if len(temp_ids) > 1:
                    temp_ids.pop()
            else:
                if len(temp_ids) > len(max_con_ids):
                    found = True
                    max_con_ids = temp_ids.copy()
                    max_con_res = copy.deepcopy(con_res)#con_res.copy()

        #removed = [i for i in lefted_ids if i in max_con_ids]

        if found:
            max_con_res.consistent_ex_ids = max_con_ids.copy()
            consistent_res.append(copy.deepcopy(max_con_res))#(max_con_res.copy())
            consistent_set_size.append(len(max_con_ids))#(len(removed) + 1) #Change to the whole max set size
            lefted_ids = [i for i in lefted_ids if i not in max_con_ids]

    if len(consistent_res)==0:
        return (0, None)

    #consistent_set_size.sort()
    data = [(consistent_set_size, consistent_res) for consistent_set_size, consistent_res in zip(consistent_set_size,consistent_res)] #先转化成元组
    data.sort(reverse=True) #按照降序排序
    consistent_set_size = [consistent_set_size for consistent_set_size,consistent_res in data] #将排好序的分数姓名的元组分开
    consistent_res = [consistent_res for consistent_set_size,consistent_res in data]

    score = 0
    for i in range(0, len(consistent_set_size)):
        score += math.exp(-i) * consistent_set_size[i]
    return (score, consistent_res[0])# the largest set


'''
# optimise the variable indicators to find the best consistent abduction of examples
def opt_var_ids(exs, maps):
    dim = Dimension(len(flatten(exs)),
                    [[0,1]] * len(flatten(exs)),
                    [False] * len(flatten(exs)))
    obj = Objective(lambda v: -consistent_score(exs, v.get_x(), maps).score, dim)
    param = Parameter(budget = 100 , autoset = True)
    solution = Opt.min(obj, param)

    return solution

# optimise the variable indicators to find the best consistent abduction of examples
def opt_var_ids_sets_constraint(exs, mapping, constraint):
    dim = Dimension(len(flatten(exs)),
                    [[0,1]] * len(flatten(exs)),
                    [False] * len(flatten(exs)))
    obj = Objective(lambda v: -consistent_score_sets(exs, [int(i) for i in v.get_x()], mapping)[0], dim = dim, constraint = constraint)
    param = Parameter(budget = 100 , autoset = True)
    solution = Opt.min(obj, param)

    return solution
'''

# count the number of chess in all examples
def num_of_chess(exs):
    l = 0
    for ex in exs:
        pos = ex[0]
        l += len(pos)
    return l

# optimise the variable indicators to find the best consistent abduction of examples
def opt_var_ids_sets_chess_constraint(exs, mapping, constraint):
    num_chess = num_of_chess(exs)
    dim = Dimension(num_chess,
                    [[0,1]] * num_chess,
                    [False] * num_chess)
    obj = Objective(lambda v: -consistent_score_sets_chess(exs, [int(i) for i in v.get_x()], mapping)[0], dim = dim, constraint = constraint)
    param = Parameter(budget = 100 , autoset = True)
    solution = Opt.min(obj, param)

    return solution
