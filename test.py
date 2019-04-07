import sys
import time
from functools import partial

sys.path.insert(0, 'src/lib/')
import LogicLayer as LL

sys.path.insert(0, 'src/python_interface/')
from python_interface import *



def test_gen_rand_feature():
    print('I. Test generating random features (addition rules)')
    print('==================\n')
    for i in range(1, 11):
        print(i)
        t = LL.genRandFeature()
        print(t)
    print('----------END------------\n')

def test_create_PlTerms():
    print('II. Test creating constants (number, lower-case letters started string/compounds) and variables (strings started with upper-case letters, strings started with \'_\' ): ')
    print('==================\n')
    t1 = LL.PlTerm()
    print(t1)
    t2 = LL.PlTerm([1, 2, 'X', 'X', 'good(day)'])
    print(t2)
    t3 = LL.PlTerm([1, 2, 'X', 'X', ['good(day)', '_d'], '_d', []])
    print(t3)
    print("*******If the term reference hasn't been messed up, the outputs stay the same")
    print(t1)
    print(t2)
    print(t3)
    print('*******Test creating list of variables')
    a = LL.PlTerm('A')
    b = LL.PlTerm([a,a,a,a,a,a])
    c = LL.PlTerm([b,b,b])
    d = LL.PlTerm(c.py())
    print('*******a: ')
    print(a)
    print('*******b & c: c and b should contain same variables as a')
    print(b)
    print(c)
    print('*******d: all d\'s elements should be same variable, but different from a')
    print(d)
    print('----------END------------\n')

def test_eval_feature():
    print('III. Test evaluate example with given feature')
    print('==================\n')
    print('*******Feature: ')
    f = LL.PlTerm(['my_op([1],[1],[1,0])','my_op([0],[0],[0])','my_op([1],[0],[1])','my_op([0],[1],[1])'])
    print(f)
    print('*******Examples: ')
    ex1 = LL.PlTerm([1,1,'+',1,1,'=',1,1])
    ex2 = LL.PlTerm([1,1,'+',1,1,'=',1,1,0])
    ex3 = LL.PlTerm(['A','A','B','A','A','C','A','A','D'])
    re1 = LL.evalInstFeature(ex1, f)
    re2 = LL.evalInstFeature(ex2, f)
    re3 = LL.evalInstFeature(ex3, f)
    print('****example 1 & result: ')
    print(ex1)
    print(re1)
    print('****example 2 & result: ')
    print(ex2)
    print(re2)
    print('****example 3 & result: ')
    print(ex3)
    print(re3)
    print('----------END------------\n')

def test_abduction():
    print('IV. Test instance abduction given feature and label, the instance should contain at least 1 variable.')
    print('==================\n')
    print('*******feature:')
    f = LL.PlTerm(['my_op([1],[1],[1,0])','my_op([0],[0],[0])','my_op([1],[0],[1])','my_op([0],[1],[1])'])
    ex1 = LL.PlTerm([1,1,'+',1,1,'=',1,1,'U']) # U for unknown
    re1 = LL.abduceInstFeature(ex1, f, True)
    re2 = LL.abduceInstFeature(ex1, f, False)
    print(f)
    print('*******ex1 & result when assuming ex1 is POSITIVE+++:')
    print('****ex1')
    print(ex1)
    print('****abduced')
    print(re1)
    print('*******result when assuming ex1 is NEGATIVE---:')
    print(re2)

    ex2 = LL.PlTerm(['U1','U2','U3','U4','_5','_6','U7','_8','U9']) # U and _ standing for unknown variables
    re2 = LL.abduceInstFeature(ex2, f, True)
    print('\n****ex2')
    print(ex2)
    print('****abduced when assuming ex2 is POSITIVE+++')
    print(re2)

    print()
    f = LL.PlTerm([])
    ex2 = LL.PlTerm(['A', '_', 'A', 'A', '_', 'A'])
    re3 = LL.abduceInstFeature(ex2, f, True)    
    print('**** abducing when feature is [] (empty):')
    print('***ex: ')
    print(ex2)
    print('***result: ')
    print(re3)

    print('----------END------------\n')

def test_parse_inst_feature():
    print('VI. Test parsing instance to feature')
    t = time.time()
    ex1 = LL.PlTerm([1,1,'+',1,1,'=',1,1])
    ex2 = LL.PlTerm([1,1,'+',1,1,'=',1,1,0])
    f1 = LL.parseInstFeature(ex1)
    f2 = LL.parseInstFeature(ex2)
    ex3 = LL.PlTerm([1,'+',1,'=',1,0])
    f3 = LL.parseInstFeature(ex3)
    print(time.time() - t)
    print(ex1)
    print('parsed to:')
    print(f1)
    print(ex2)
    print('parsed to:')    
    print(f2)
    print(ex3)
    print('parsed to:')    
    print(f3)
    print('----------END------------\n')

def test_legit_inst():    
    print('VI. Test veryfication of legitimate instance')
    ex = []
    ex.append(LL.PlTerm([1,1,'+',1,1,'=',1,1]))
    ex.append(LL.PlTerm([1,1,'+',1,1,'=',1,1,0,'+']))
    ex.append(LL.PlTerm([1,1,'+',1,1,'=',1,1,'U']))
    ex.append(LL.PlTerm([1,'+','+',1,1,'=',1,1]))
    t = time.time()
    for i in range(0, len(ex)):
        print(ex[i])
        print(LL.legitInst(ex[i]))
    print(time.time() - t)
    print('----------END------------\n')

def test_consistent_abduce():
    print('VII. Test consistent abduction')
    ex1_1 = LL.PlTerm([1,1,'_',1,'_','_',1,'_'])
    ex1_2 = LL.PlTerm([1,1,'_',1,'_','_',0,'_'])
    ex1_3 = LL.PlTerm([1,'_','_',0,'_','_','_','_'])
    ex2_1 = LL.PlTerm(['_','_','_','_','_'])
    ex2_2 = LL.PlTerm([0,'_',0,'_','_','_'])
    ex3 = LL.PlTerm([1,'_','_','_',1,0])
    ex4 = LL.PlTerm([1,'_',1,1,'_',1,'_','_'])
    ex5 = LL.PlTerm([1,'+',0,'=',1])
    ex6 = LL.PlTerm([1,1,'+',1,1,'=',1,1,0])
    ex7 = LL.PlTerm([0,'+',1,'=','_'])
    exs1 = LL.PlTerm([ex1_1, ex2_1, ex3, ex4, ex5, ex6, ex7])
    exs2 = LL.PlTerm([ex1_2, ex2_1, ex3, ex4, ex5, ex6, ex7])
    exs3 = LL.PlTerm([ex1_3, ex2_2, ex3, ex4, ex5, ex6, ex7])
    exs4 = LL.PlTerm([ex1_2, ex3, ex4])
    exs5 = LL.PlTerm([[1, 1, '+', 1, 1, '=', 1, 0], [0, '+', 0, '=', 0], [1, '+', 1, '=', 1, 0], [1, '+', 1, 1, '=', 1, 0, 0], [1, '+', 0, '=', 1]])
    t = time.time()
    re1 = LL.abduceConInsts(exs1)
    re2 = LL.abduceConInsts(exs2)
    re3 = LL.abduceConInsts(exs3)
    re4 = LL.abduceConInsts(exs4)
    re5 = LL.abduceConInsts(exs5)
    print(time.time() - t)
    print('*****ex1:')
    print(exs1)
    print('***result1:')
    if re1:
        print(re1.py())
    else:
        print(re1)
    print('*****ex2:')
    print(exs2)
    print('***result2:')
    if re2:
        print(re2.py())
    else:
        print(re2)
    print('*****ex3:')
    print(exs3)
    print('***result3:')
    if re3:
        print(re3.py())
    else:
        print(re3)
    print('*****ex4:')
    print(exs4)
    print('***result4:')
    if re4:
        print(re4.py())
    else:
        print(re4)
    print('*****ex5:')
    print(exs5)
    print('***result5:')
    if re5:
        print(re5.py())
    else:
        print(re5)
        
    print('----------END------------\n')

def test_consistent_score():
    print('VIII. Test consistent score')
    ex1_1 = [3,3,3,3,3,3,3,3]
    ex1_2 = [3,3,3,3,3,3,4,3]
    ex1_3 = [3,3,3,4,3,3,4,3]
    var1_1 = [0,0,1,0,1,1,0,1]
    var1_2 = [0,0,1,0,1,1,0,1]
    var1_3 = [0,1,1,0,1,1,1,1]
    ex2_1 = [3,2,3,3,3]
    var2_1 = [1,1,1,1,1]
    ex2_2 = [4,3,4,3,3,3]
    var2_2 = [0,1,0,1,1,1]
    ex3 = [3,3,3,3,3,4]
    var3 = [0,1,1,1,0,0]
    ex4 = [3,3,3,3,3,3,3,3]
    var4 = [0,1,0,0,1,0,1,1]
    ex5 = [3,2,4,1,3]
    var5 = [0,0,0,0,0]
    ex6 = [3,3,2,3,3,1,3,3,4]
    var6 = [0,0,0,0,0,0,0,0,0]
    ex7 = [4,2,3,1,3]
    var7 = [0,0,0,0,1]
    exs1 = [ex1_1, ex2_1, ex3, ex4, ex5, ex6, ex7]
    exs2 = [ex1_2, ex2_1, ex3, ex4, ex5, ex6, ex7]
    exs3 = [ex1_3, ex2_2, ex3, ex4, ex5, ex6, ex7]
    exs4 = [ex1_2, ex3, ex4]
    vars1 = flatten([var1_1, var2_1, var3, var4, var5, var6, var7])
    vars2 = flatten([var1_2, var2_1, var3, var4, var5, var6, var7])
    vars3 = flatten([var1_3, var2_2, var3, var4, var5, var6, var7])
    vars4 = flatten([var1_2, var3, var4])

    maps = gen_mappings([0,1,2,3,4], ['+','=',0,1,'null'])
    t = time.time()
    result1 = consistent_score(exs1, vars1, maps)
    print(time.time() - t)
    t = time.time()
    result2 = consistent_score(exs2, vars2, maps)
    print(time.time() - t)
    t = time.time()
    result3 = consistent_score(exs3, vars3, maps)
    print(time.time() - t)
    t = time.time()
    result4 = consistent_score(exs4, vars4, maps)
    print(time.time() - t)
    print('*****ex1:')
    print('input:', end = '\t')
    print(exs1)
    print('vars:', end = '\t')
    print(vars1)
    print('***result1:')
    print('score:', end = ' ')
    print(result1.score, end = " -> ")
    print(result1.abduced_exs)
    print('*****ex2:')
    print('input:', end = '\t')
    print(exs2)
    print('vars:', end = '\t')
    print(vars2)
    print('***result2:')
    print('score:', end = ' ')
    print(result2.score, end = " -> ")
    print(result2.abduced_exs)
    print('*****ex3:')
    print('input:', end = '\t')
    print(exs3)
    print('vars:', end = '\t')
    print(vars3)
    print('***result3:')
    print('score:', end = ' ')
    print(result3.score, end = " -> ")
    print(result3.abduced_exs)
    print('*****ex4:')
    print('input:', end = '\t')
    print(exs4)
    print('vars:', end = '\t')
    print(vars4)
    print('***result4:')
    print('score:', end = ' ')
    print(result4.score, end = " -> ")
    print(result4.abduced_exs)
    print('----------END------------\n')

def test_opt_consistent_score():
    print('IX. Test consistent score optimisation')
    ex1_1 = [3,3,3,3,3,3,3,3]
    ex1_2 = [3,3,3,3,3,3,4,3]
    ex1_3 = [3,3,3,4,3,3,4,3]
    ex2_1 = [3,2,3,3,3]
    ex2_2 = [4,3,4,3,3,3]
    ex3 = [3,3,3,3,3,4]
    ex4 = [3,3,3,3,3,3,3,3]
    ex5 = [3,2,4,1,3]
    ex6 = [3,3,2,3,3,1,3,3,4]
    ex7 = [4,2,3,1,3]
    exs1 = [ex1_1, ex2_1, ex3, ex4, ex5, ex6, ex7]
    exs2 = [ex1_2, ex2_1, ex3, ex4, ex5, ex6, ex7]
    exs3 = [ex1_3, ex2_2, ex3, ex4, ex5, ex6, ex7]
    exs4 = [ex1_2, ex3, ex4]

    maps = gen_mappings([0,1,2,3,4], ['+','=',0,1,'null'])


    print("*************ex4")
    print(exs4)
    t = time.time()
    sol = opt_var_ids(exs4, maps)
    print('elapsed time: ', end = '')
    print(time.time() - t)
    print(sol.get_x())
    result4 = consistent_score(exs4, sol.get_x(), maps)
    print(result4.score, end = " ---> ")
    print(result4.abduced_exs)
    print(result4.abduced_map)
    print(result4.abduced_rules)

    print("*************ex3")
    print(exs3)
    t = time.time()
    sol = opt_var_ids(exs3, maps)
    print('elapsed time: ', end = '')
    print(time.time() - t)
    print(sol.get_x())
    result3 = consistent_score(exs3, sol.get_x(), maps)
    print(result3.score, end = " ---> ")
    print(result3.abduced_exs)
    print(result3.abduced_map)
    print(result3.abduced_rules)

    print("*************ex2")
    print(exs2)
    t = time.time()
    sol = opt_var_ids(exs2, maps)
    print('elapsed time: ', end = '')
    print(time.time() - t)
    print(sol.get_x())
    result2 = consistent_score(exs2, sol.get_x(), maps)
    print(result2.score, end = " ---> ")
    print(result2.abduced_exs)
    print(result2.abduced_map)
    print(result2.abduced_rules)

    print("*************ex1")
    print(exs1)
    t = time.time()
    sol = opt_var_ids(exs1, maps)
    print('elapsed time: ', end = '')
    print(time.time() - t)
    print(sol.get_x())
    result1 = consistent_score(exs1, sol.get_x(), maps)
    print(result1.score, end = " ---> ")
    print(result1.abduced_exs)
    print(result1.abduced_map)
    print(result1.abduced_rules)

    print('----------END------------\n')

def constraint(solution, min_var, max_var):
    x = solution.get_x()
    #print (x,max_var-x[0,:].sum())
    return (max_var - x.sum())*(x.sum() - min_var)


# 最终NLM需要用到的一致性测试
def final_test():
    ex1_1 = [3,3,3,3,3,3,3,3]
    ex1_2 = [3,3,3,3,3,3,4,3]
    ex1_3 = [3,3,3,4,3,3,4,3]
    ex2_1 = [3,2,3,3,3]
    ex2_2 = [4,3,4,3,3,3]
    ex3 = [3,3,3,3,3,4]
    ex4 = [3,3,3,3,3,3,3,3]
    ex5 = [3,2,4,1,3]
    ex6 = [3,3,2,3,3,1,3,3,4]
    ex7 = [4,2,3,1,3]
    exs1 = [ex1_1, ex2_1, ex3, ex4, ex5, ex6, ex7]
    exs2 = [ex1_2, ex2_1, ex3, ex4, ex5, ex6, ex7]
    exs3 = [ex1_3, ex2_2, ex3, ex4, ex5, ex6, ex7]
    exs4 = [ex1_2, ex3, ex4]
    #exs = [[0, 1, 2, 1, 0, 3, 1, 0, 1], [1, 1, 2, 1, 3, 1, 0, 0], [1, 1, 2, 1, 1, 1, 3, 1, 0, 1, 0], [0, 2, 0, 3, 0],[1, 1, 2, 1, 0, 3, 1, 0, 1], [1, 1, 2, 1, 3, 1, 0, 0], [1, 1, 2, 1, 1, 1, 3, 1, 0, 1, 0], [0, 2, 0, 3, 0],[1, 1, 2, 1, 0, 3, 1, 0, 1], [1, 1, 2, 1, 3, 1, 0, 0], [1, 1, 2, 1, 1, 1, 3, 1, 0, 1, 0], [0, 2, 0, 3, 0]]
    #exs = [[4, 2, 4, 1, 2, 3, 2, 0, 1], [2, 1, 2, 4, 1, 2, 3, 1, 4, 2, 4], [1, 2, 1, 1, 4, 1, 0, 2, 1, 1, 1], [2, 2, 3, 1, 2, 3, 4, 3, 2, 1, 1, 2]]

    #exs = exs3
    # bad case 1
    # exs = [[1, 0, 3, 2, 3], [3, 1, 0, 3, 2, 3, 3], [3, 0, 1, 2, 3], [3, 0, 3, 2, 3, 1], [1, 0, 3, 2, 3], [3, 0, 3, 1, 2, 3, 3], [1, 0, 3, 1, 2, 3, 1], [3, 0, 1, 2, 3], [3, 0, 1, 2, 3], [3, 0, 3, 2, 3, 1], [3, 0, 1, 2, 3], [1, 0, 3, 2, 3], [1, 0, 3, 2, 3], [3, 0, 1, 2, 3], [1, 0, 1, 2, 1], [3, 0, 3, 2, 3, 1], [3, 0, 1, 2, 3], [3, 0, 3, 2, 3, 1], [3, 3, 0, 1, 2, 3, 3], [3, 0, 3, 2, 3, 1]]
    # bad case 2
    #exs = [[2, 3, 0, 1, 2], [2, 3, 2, 1, 2, 0], [0, 3, 0, 1, 0], [2, 3, 0, 1, 2], [2, 3, 2, 1, 2, 0], [2, 3, 2, 1, 2, 0], [0, 3, 2, 1, 2], [2, 3, 2, 1, 2, 0], [2, 3, 2, 0, 1, 2, 2], [2, 0, 3, 2, 1, 2, 2], [2, 3, 2, 1, 2, 0], [2, 3, 0, 1, 2], [2, 2, 3, 0, 1, 2, 2], [2, 3, 2, 1, 2, 0], [2, 3, 2, 1, 2, 0], [2, 3, 2, 1, 2, 0], [0, 3, 2, 1, 2], [2, 3, 0, 1, 2], [2, 3, 2, 1, 2, 0], [2, 3, 0, 1, 2]]
    #maps = gen_mappings([0,1,2,3,4], ['+','=',0,1,'null'])


    exs = [[1,2,1,3,1,0],[1,2,0,3,1]] # CNN输出的等式字符label序列
    exs =[[0, 2, 1, 0, 3, 1, 0], [1, 0, 2, 1, 2, 1, 1], [0, 2, 1, 0, 3, 1, 0], [0, 2, 1, 1, 3, 1, 1], [1, 0, 2, 0, 3, 1, 1]]
    exs = [[0, 2, 0, 3, 0], [0, 2, 1, 3, 1], [0, 2, 1, 3, 0], [0, 2, 1, 3, 1]]

    #0+10=10 10+1+11 0+10=10 0+11=11 10+0=11
    maps = gen_mappings([0,1,2,3], ['+','=',0,1]) # 所有可能从label到符号的mappings

    print(exs)
    
    c = partial(constraint, min_var = 2, max_var = 6) # 约束zoopt最少只修改0个比特，最多修改5个比特

    out_eqs = [] # 经过abduction修正以后，输出给CNN重训的等式集合

    # 对所有的mappings进行迭代
    for m in maps:
        #m = {0: 'null', 1: 1, 2: 0, 3: '=', 4: '+'}
        #m = {0: '+', 1: 0, 2: '=', 3: 1, 4: 'null'}
        m = {0: 0, 1: 1, 2: '+', 3: '='} # 只尝试这一种映射
        print(m)
        t = time.time()

        # 根据输入等式集合exs、符号映射m和约束c，进行一致性优化；
        # 该函数返回一个var_id（标记哪些未知该被修改）序列，
        # 该函数的目标是通过优化var_id，找到exs里最大的一个子集，使得该子集通过var_id标记后，
        # abduction能够对被标记的符号进行修改，使得该子集中的等式一致。
        # 其中的子集选择通过贪心搜索来找到
        sol = opt_var_ids_sets_constraint(exs, m, c)

        # 将得到的最优var_id应用在exs上，并abduce出最大一致的等式子集的label（用于重训CNN）
        # REMARK：这里必须再次使用consistent_score_sets进行abduction，因为racos只能返回二进制
        #         向量var_ids. 因此要用con_score_sets重新将var_ids转化为等式out_eqs
        #         和规则集feature
        consist_res = consistent_score_sets(exs, [int(i) for i in sol.get_x()], m)
        print('time:', end = '\t')
        print(time.time() - t)
        print(consist_res)

        # con_score_sets返回的consist_res=(score, eq_sets). 其中score是给zoopt用的函数值，
        # 所以打印结果时只用管它的第2个元素
        for consist_re in consist_res[1]:
            print('****Consistent instance:')
            print('consistent examples:', end = '\t')
            print(consist_re.consistent_ex_ids) # 最大一致子集的元素在原本exs中的序列号
            print('mapping:', end = '\t')
            print(consist_re.abduced_map) # 反绎推理的到的Mapping（如果不考虑同mapping就无所谓了）
            print('abduced examples:', end = '\t')
            print(consist_re.abduced_exs) # 经过反绎推理后，被修改用于重训CNN的label序列
            print('consistent equation set:', end = '\t')
            print(consist_re.abduced_rules) # 被修正后的label序列经过mapping翻译成等式符号的样子

            # 将这个consistent的结果转化为加法规则my_op和mappings共同构成的feature，
            # 用于训练决策网络（MLP）
            feat = consist_re.to_feature()
            print('****Learned feature:')
            print('mapping: ', end = '\t')
            print(feat.mapping)
            print('rules: ', end = '\t')
            rule_set = feat.rules.py();
            print(rule_set)
            #print(feat.eval([1, 2, 2, 4, 1, 1, 3, 1, 1, 1, 0]))
            if (len(rule_set) > 0):
                out_eqs.append(rule_set)
        break
    return out_eqs

if __name__ == "__main__":

    LL.init("-G10g -M6g") # must initialise prolog engine first!
    LL.consult('src/prolog/learn_add.pl') # consult the background knowledge file
    #LL.call("current_prolog_flag(table_space, X), writeln(X)")  #test if stack changed
    LL.call("prolog_stack_property(global, limit(X)), writeln(X)")  #test if stack changed
    
    
    test_gen_rand_feature() # test random feature generation
    test_create_PlTerms() # test PlTerm creation, including variables and constants
    test_eval_feature() # test evaluating examples with feature
    

    test_legit_inst() # test evaluation of legitimate instance mapping
    test_parse_inst_feature() # test parsing instance to feature
    
    test_consistent_abduce() # test consistent abduction for a list of instances

    test_consistent_score() # test greedy consistent score function
    
    test_opt_consistent_score() # test consistent score optimisation with zoopt

    '''
    rules = final_test()
    print('my_op Rules: ')
    print(rules)
    print(LL.conDigitRules(rules[0]))
    print(LL.evalInstRules([1, '+', 1, 0, '=', 1, 1], rules[0]))
    print(LL.evalInstRules([1, '+', 1, 0, '=', 1, 1], ['my_op([1],[0],[1])', 'my_op([0],[1],[1])']))
    print(LL.evalInstRules([1, '+', 1, 1, '=', 1, 1], ['my_op([1],[0],[1])', 'my_op([0],[1],[1])']))
    print(LL.evalInstRules([1, '+', 1, 1, '=', 1, 1], ['my_op([1],[1],[1])']))
    '''

    '''
    for i in range(0, 100):
        test_consistent_score()
        # LL.gc()
        # LL.trimStacks()

    for i in range(0, 10000):
        t = time.time()
        f = LL.PlTerm(['my_op([1],[1],[1,0])','my_op([0],[0],[0])','my_op([1],[0],[1])','my_op([0],[1],[1])'])
        ex = LL.PlTerm(['A','A','B','A','A','C','A','A','D'])
        print(LL.evalInstFeature(ex, f))
        print(LL.PlTerm('_'))
        print (time.time() - t)
    '''
    # test_abduction() # test abduction
    #LL.halt() # clean prolog stack
