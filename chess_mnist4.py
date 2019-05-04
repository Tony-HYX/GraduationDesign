import os
import numpy as np
import random
random_seed = 7#random.randint(0, 10000)
print("select random seed is : ", random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
import keras
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras import optimizers
from models import NN_model
from PIL import Image
from functools import partial

import sys
sys.path.insert(0, 'src/lib/')
import LogicLayer as LL
sys.path.insert(0, 'src/python_interface/')
from python_interface import *


def get_img_data(src_path, labels, shape = (28, 28, 1)):
    print("Start getting img data from the source")
    X = [] #image
    Y = [] #label
    h = shape[0]
    w = shape[1]
    d = shape[2]
    for (index,label) in enumerate(labels):  #index = [0,1,2,3]
        label_folder_path = os.path.join(src_path,label)
        for p in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path,p)
            #print(image_path)
            if d == 1:
                mode = 'I'
            else:
                mode = 'RGB'
            image = Image.open(image_path).convert(mode).resize((h, w))
            X.append((np.array(image)-127)*(1/128.))
            Y.append(index)
    
    X = np.array(X)
    Y = np.array(Y)
    
    index = np.array(list(range(len(X))))
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]
    
    assert (len(X) == len(Y))
    print("Total data size is :", len(X))

    X = X.reshape(-1, h, w, d)      # normalize
    Y = np_utils.to_categorical(Y, num_classes=len(labels))
    return X, Y

def get_nlm_net(labels_num, shape = (28, 28, 1), model_name = "LeNet5"):
    assert model_name == "LeNet5"
    if model_name == "LeNet5":
        return NN_model.get_LeNet5_net(labels_num, shape)
    '''
    d = shape[2]
    if d==1:
        return NN_model.get_LeNet5_net(labels_num, shape)
    else:
        return NN_model.get_cifar10_net(labels_num, shape)
    '''

def net_model_test(src_path, labels, src_data_name, shape = (28, 28, 1)):
    file_name = '%s_correct_model_weights.hdf5'%src_data_name
    if os.path.exists(file_name):
        print("Model file exists")
        return
    X, Y = get_img_data(src_path, labels, shape)

    # train:test = 5:1
    X_train = X[:len(X)//5*1]
    y_train = Y[:len(Y)//5*1]
    X_test = X[len(X)//5*1:]
    y_test = Y[len(Y)//5*1:]
    
    print("Start training the correct model...")
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    model = get_nlm_net(len(labels), shape)
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6) #Add decay
    model.compile(optimizer=opt_rms, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('Training')
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test,y_test))
    model.save_weights(file_name)

    print('\nTesting')
    loss, accuracy = model.evaluate(X_test, y_test)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

def net_model_pretrain(src_path, labels, src_data_name, shape= (28, 28, 1)):
    file_name = '%s_pretrain_weights.hdf5'%src_data_name
    h = shape[0]
    w = shape[1]
    d = shape[2]
    if os.path.exists(file_name):
        print("Pretrain file exists")
        return
    
    X, Y = get_img_data(src_path, labels, shape)
    print("\nThere are %d pretrain images"%len(X))

    Y = X.copy().reshape(-1, h * w *d)
    model = NN_model.get_mnist_autoencoder_net(len(labels), input_shape = shape)
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    
    print('Pretraining...')
    model.fit(X, Y, epochs=10, batch_size=64)
    model.save_weights(file_name)
    print("Pretrain ended")

def current_model_accuracy(model, src_path, labels, input_shape, abduced_map):
    #model = NN_model.get_LeNet5_net(len(labels), shape)
    #model.load_weights('mnist_images_correct_model_weights.hdf5')
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\nEvaluating current model's perception accuracy...")
    X, Y = get_img_data(src_path, labels, input_shape)
    for mapping in gen_mappings([0,1,2], [0,1,2]):
        print('\n',mapping)
        real_Y = []
        correct_cnt = np.array([0]*len(labels))
        label_cnt = np.array([0]*len(labels))
        for y in Y:
            real_Y.append(mapping[np.argmax(y)]) #abduced_map
        
        predict_Y = np.argmax(model.predict(X), axis=1)
        for (real_y,predict_y) in zip(real_Y,predict_Y):
            if real_y==predict_y:
                correct_cnt[real_y] += 1
            label_cnt[real_y] += 1
        print(correct_cnt)
        print(label_cnt)
        print('Test accuracy: ', correct_cnt.sum()/label_cnt.sum())
            

def LL_init(pl_file_path = "src/prolog/learn.chr"):
    print("Initializing prolog...")
    assert os.path.exists(pl_file_path), "%s is not exist" % pl_file_path
    LL.init("-G10g -M6g") # must initialise prolog engine first!
    LL.consult(pl_file_path) # consult the background knowledge file
    LL.call("prolog_stack_property(global, limit(X)), writeln(X)")  #test if stack changed
    print("Prolog has alreadly initialized.")

def divide_chessboards_by_len(chessboards):
    '''Divide chessboards by number of chess'''
    # Sort
    chessboards.sort(key = lambda i:len(i)) 
    
    chessboards_by_len = list()
    start = 0
    for i in range(1,len(chessboards)+1):
        #print(len(chessboards[i]))
        if i==len(chessboards) or len(chessboards[i])!=len(chessboards[start]):
            chessboards_by_len.append(chessboards[start:i])
            start = i
    '''
    for chessboards in chessboards_by_len:
        for chessboard in chessboards:
            for chess in chessboard:
                print(chess[0],chess[1])
            print(len(chessboard),'\n')
    '''
    return chessboards_by_len

def split_chessboard(chessboards_by_len,prop_train,prop_val):
    train = []
    val = []
    all_prop = prop_train + prop_val
    for chessboards in chessboards_by_len:
        train.append(chessboards[:len(chessboards)//all_prop*prop_train])
        val.append(chessboards[len(chessboards)//all_prop*prop_train:])
        #print(len(chessboards[:len(chessboards)//all_prop*prop_train]))
        #print(len(chessboards[len(chessboards)//all_prop*prop_train:len(chessboards)//all_prop*(prop_train+prop_val)]))
        #print(len(chessboards[len(chessboards)//all_prop*(prop_train+prop_val):]))
    return train, val

def constraint(solution, min_var, max_var):
    x = solution.get_x()
    #print (x,max_var-x[0,:].sum())
    return (max_var - x.sum())*(x.sum() - min_var)

def get_chess_img_or_label(chessboards, ids = None): # use the same function because they are in the same position && chessboard=[chesses, true_or_false_label]
    img_data = []
    if ids is None:
        for chessboard in chessboards:
            for chess in chessboard[0]: #chessboard[1] is label
                img_data.append(chess[2]) 
    else:
        for id in ids:
            chessboard = chessboards[id]
            for chess in chessboard[0]:
                img_data.append(chess[2]) #chessboard[1] is label
    img_data = np.array(img_data)
    return img_data
    
def get_format_data_from_model(model, chessboards): # chessboard=[chesses, true_or_false_label]
    labels = np.argmax(model.predict(get_chess_img_or_label(chessboards)), axis=1)
    cnt = 0
    
    #Prepare format data from label
    exs = []
    for chessboard in chessboards:
        ex = []
        for chess in chessboard[0]:
            e = []
            e.append(chess[0])
            e.append(chess[1])
            e.append(labels[cnt])
            cnt += 1
            ex.append(e)
        exs.append([ex,[],chessboard[1]])
    assert(cnt==len(labels))
    return exs
    
def translate_image_to_label(model, chessboards, mapping): # chessboard has no true or false label   use for valuation data
    cnt = 0
    exs = []
    for chessboard in chessboards:
        ex = []
        for chess in chessboard:
            label = np.argmax(model.predict(np.array([chess[2]])))
            #label = np.random.randint(0, 3)
            e = [chess[0], chess[1], mapping[label]]
            ex.append(e)
        exs.append(ex)
    return exs

def evaluate_data(model, chessboards_true, chessboards_false, mapping):
    correct_cnt_true = 0
    correct_cnt_false = 0
    exs_true = translate_image_to_label(model, chessboards_true, mapping) 
    exs_false = translate_image_to_label(model, chessboards_false, mapping) 
    for ex in exs_true:
        re = LL.evalChess(LL.PlTerm(ex))
        if re==True:
            correct_cnt_true += 1
    for ex in exs_false:
        re = LL.evalChess(LL.PlTerm(ex))
        if re==False:
            correct_cnt_false += 1
    #Get MLP validation result
    if len(exs_true) != 0:
        true_accuracy = correct_cnt_true / len(exs_true)
    else:
        true_accuracy = 99
    if len(exs_false) != 0:
        false_accuracy = correct_cnt_false / len(exs_false)
    else:
        false_accuracy = 99
    total_accuracy = (correct_cnt_true+correct_cnt_false) / (len(exs_true)+len(exs_false))
    return true_accuracy, false_accuracy, total_accuracy

def get_abduced_chessboards_labels(model, chessboards, maps = None, for_mlp_vector = False, shape = (28, 28, 1)):
    # Get the model's output
    h = shape[0]
    w = shape[1]
    d = shape[2]
    abduced_label = None
    exs = []
    consistent_ex_ids = None
    
    print("\n\nThis is the model's label before abduce:")
    exs = get_format_data_from_model(model, chessboards)
    print(exs)
    
    if maps is None:
        maps = gen_mappings([0,1,2], [0,1,2]) # 所有可能的mappings

    # Check if it can abduce rules without changing any labels
    consist_res_max = (-1, None)
    ret_map = None
    for m in maps:
        consist_res = consistent_score_sets_chess(exs, [0]*num_of_chess(exs), m)  #Assuming that each equation is the same length
        if consist_res[0] > consist_res_max[0]:
            consist_res_max = consist_res
            ret_map = m
        #input()############

    if consist_res_max[1]:
        consist_re = consist_res_max[1]
        if len(consist_re.consistent_ex_ids)==len(chessboards):
            print("#It can abduce rules without changing any labels")
            #abduced_label = np_utils.to_categorical(flatten(consist_re.abduced_exs), num_classes=len(labels))
            return (consist_re.consistent_ex_ids, consist_re.abduced_exs, ret_map, [])
    if for_mlp_vector and consist_res_max[1] is None: #If rules are used in mlp vector, it cannnot change any label
        return (None, None, None, None)
    
    # Find the possible wrong position in symbols and Abduce the right symbol through logic module
    c = partial(constraint, min_var = 1, max_var = 10) # 约束zoopt最少只修改1个比特，最多修改10个比特
    for m in maps:
        sol = opt_var_ids_sets_chess_constraint(exs, m, c)
        
        consist_res = consistent_score_sets_chess(exs, [int(i) for i in sol.get_x()], m)
    
        if consist_res[0] > consist_res_max[0]:
            consist_res_max = consist_res
            ret_map = m
            
    # con_score_sets返回的consist_res=(score, eq_sets). 其中score是给zoopt用的函数值，所以打印结果时只用管它的第2个元素
    if consist_res_max[1] is None:  #Cannot abduce
        return (None, None, None, None)
    consist_re = consist_res_max[1]
            
    print('****Consistent instance:')
    print('consistent examples:', end = '\t')
    print(consist_re.consistent_ex_ids) # 最大一致子集的元素在原本exs中的序列号
    print('mapping:', end = '\t')
    print(consist_re.abduced_map) # 反绎推理的到的Mapping
    print('abduced examples:', end = '\t')
    print(consist_re.abduced_exs) # 经过反绎推理后，被修改用于重训CNN的label序列
    #print('consistent equation set:', end = '\t')
    #print(consist_re.abduced_rules) # 被修正后的label序列经过mapping翻译成等式符号的样子

    # 将这个consistent的结果转化为加法规则my_op和mappings共同构成的feature，
    # 用于训练决策网络（MLP）
    '''
    feat = consist_re.to_feature()
    print('****Learned feature:')
    print('rules: ', end = '\t')
    rule_set = feat.rules.py();
    print(rule_set)
    '''

    # Convert the symbol to network output
    #abduced_label = np_utils.to_categorical(flatten(abduced_label), num_classes=len(labels))
    return (consist_re.consistent_ex_ids, consist_re.abduced_exs, ret_map , []) # rule_set)


def get_mlp_vector(equation, model, rules, abduced_map, shape=(28, 28, 1)):
    h = shape[0]
    w = shape[1]
    d = shape[2]
    model_output = np.argmax(model.predict(equation.reshape(-1, h, w, d)), axis=1)
    model_labels = []
    for out in model_output:
        model_labels.append(abduced_map[out])
    #print(model_labels)
    vector = []
    for rule in rules:
        ex = LL.PlTerm(model_labels)
        f = LL.PlTerm(rule)
        if LL.evalInstFeature(ex, f):
            vector.append(1)
        else:
            vector.append(0)
    #print(vector)
    return vector
        
def get_mlp_data(chessboards_true, chessboards_false, base_model, out_rules, abduced_map):
    mlp_vectors = []
    mlp_labels = []
    for equation in chessboards_true:
        mlp_vectors.append(get_mlp_vector(equation, base_model, out_rules, abduced_map, (32, 32, 3)))
        mlp_labels.append(1)
    for equation in chessboards_false:
        mlp_vectors.append(get_mlp_vector(equation, base_model, out_rules, abduced_map, (32, 32, 3)))
        mlp_labels.append(0)
    mlp_vectors = np.array(mlp_vectors)
    mlp_labels = np.array(mlp_labels)
    return mlp_vectors, mlp_labels

def random_select_chessboards(chessboards_true, chessboards_false, SELECT_NUM):
    select_chessboards = []
    select_chessboards_false = []
    
    if len(chessboards_true)>0:
        select_index = np.random.randint(len(chessboards_true), size=SELECT_NUM)
        select_index = np.unique(select_index)
        for idx in select_index:
            select_chessboards.append([chessboards_true[idx], True])
    
    if len(chessboards_false)>0:
        select_index = np.random.randint(len(chessboards_false), size=SELECT_NUM)
        select_index = np.unique(select_index)
        for idx in select_index:
            select_chessboards_false.append([chessboards_false[idx], False])
        
    select_chessboards.extend(select_chessboards_false)
    random.shuffle(select_chessboards)
    return select_chessboards

def nlm_main_func(labels, src_data_name, src_data_file, shape = (28, 28, 1)):
    h = shape[0]
    w = shape[1]
    d = shape[2]
    LL_init("src/prolog/learn.chr")
    SELECT_NUM = 5 #Select 5+5 chessboard to abduce rules
    
    with open(src_data_file, 'rb') as f:
        chessboards=pickle.load(f)
    input_file_true = chessboards['train:positive']
    input_file_false = chessboards['train:negative']
    input_file_true_test = chessboards['test:positive'] 
    input_file_false_test = chessboards['test:negative']

    
    t_model = NN_model.get_mnist_autoencoder_net(len(labels), shape)
    t_model.load_weights('%s_pretrain_weights.hdf5'%src_data_name)
    
    correct_model = get_nlm_net(len(labels), shape, "LeNet5")
    correct_model.load_weights('mnist_images_correct_model_weights.hdf5')
    
    base_model = get_nlm_net(len(labels), shape, "LeNet5")
    for i in range(len(base_model.layers)):
        base_model.layers[i].set_weights(t_model.layers[i].get_weights())
        
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    base_model.compile(optimizer=opt_rms, loss='categorical_crossentropy', metrics=['accuracy'])
    #base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    current_model_accuracy(base_model, './dataset/mnist_images', labels, shape, {0:0,1:1,2:2})
    
    chessboards_true_by_len = divide_chessboards_by_len(input_file_true)
    chessboards_false_by_len = divide_chessboards_by_len(input_file_false)
    chessboards_true_by_len_test = divide_chessboards_by_len(input_file_true_test)
    chessboards_false_by_len_test = divide_chessboards_by_len(input_file_false_test)

    #train:validation = 3:1
    chessboards_true_by_len_train, chessboards_true_by_len_validation = split_chessboard(chessboards_true_by_len, 3, 1) 
    chessboards_false_by_len_train, chessboards_false_by_len_validation = split_chessboard(chessboards_false_by_len, 3, 1) 
    
    for chessboards_true in chessboards_true_by_len:
        print("There are %d true training and validation chessboards of size %d"%(len(chessboards_true),len(chessboards_true[0])))
    for chessboards_false in chessboards_false_by_len:
        print("There are %d false training and validation chessboards of size %d"%(len(chessboards_false),len(chessboards_false[0])))
    for chessboards_true in chessboards_true_by_len_test:
        print("There are %d true testing chessboards of size %d"%(len(chessboards_true),len(chessboards_true[0])))
    for chessboards_false in chessboards_false_by_len_test:
        print("There are %d false testing chessboards of size %d"%(len(chessboards_false),len(chessboards_false[0])))
    
    '''
    # Test dataset
    for course_num in range(0, max(len(chessboards_true_by_len),len(chessboards_false_by_len)) ): #for each size of chessboards
        print("Course number:",course_num)
        for mapping in gen_mappings([0,1,2], [0,1,2]):
            print(mapping)
            true_accuracy,false_accuracy,total_accuracy = evaluate_data(correct_model, chessboards_true_by_len[course_num], chessboards_false_by_len[course_num], mapping) #mapping
            print(true_accuracy,false_accuracy,total_accuracy)
    input()
    '''
    
    abduced_map = None
    for course_num in range(0, max(len(chessboards_true_by_len),len(chessboards_false_by_len)) ): #for each size of chessboards
        # Prevent invalid index
        if course_num>=len(chessboards_true_by_len_train):
            chessboards_true = []
        else:
            chessboards_true = chessboards_true_by_len_train[course_num]
        if course_num>=len(chessboards_false_by_len_train):
            chessboards_false = []
        else:
            chessboards_false = chessboards_false_by_len_train[course_num]######################################
        if course_num>=len(chessboards_true_by_len_validation):
            chessboards_true_val = []
        else:
            chessboards_true_val = chessboards_true_by_len_validation[course_num]
        if course_num>=len(chessboards_false_by_len_validation):
            chessboards_false_val = []
        else:
            chessboards_false_val = chessboards_false_by_len_validation[course_num]
        # If cannot learn or validate
        if (len(chessboards_true)==0 and len(chessboards_false)==0) \
          or (len(chessboards_true_val)==0 and len(chessboards_false_val)==0):
            continue
            
        condition_cnt = 0  #the times that the condition of beginning to evaluate is continuously satisfied
        accuracy = 0  #accuracy of evaluation
        while True: 
            #Randomly select several chessboards
            select_chessboards = random_select_chessboards(chessboards_true, chessboards_false, SELECT_NUM)
            if course_num >= 2:
                abduced_map = [abduced_map]
            else:
                abduced_map = None
            
            consistent_ex_ids, abduced_chessboards, abduced_map, _ = get_abduced_chessboards_labels(base_model, select_chessboards, abduced_map, False, shape)

            if abduced_chessboards is None: #Failed
                continue
            #print(abduced_chessboards)
            
            train_pool_X = np.array(get_chess_img_or_label(select_chessboards, consistent_ex_ids))
            abduced_labels = np.array(get_chess_img_or_label(abduced_chessboards))
            train_pool_Y = np_utils.to_categorical(abduced_labels, num_classes=len(labels))
            print("\nTrain pool size is :", len(train_pool_X))

            # Train NN
            print("Training...")
            base_model.fit(train_pool_X, train_pool_Y, batch_size=32, epochs=10, verbose=0)

            print("The abduced map is:", abduced_map)
            
            print("This is the abduced label (after using map):")
            for chessboard in abduced_chessboards:
                label_mapped = apply_mapping_chess(chessboard, abduced_map)
                print(label_mapped, end=' ')
            
            #print("\nThis is the label of model after abduce(after using map):")
            model_labels = []
            for id in consistent_ex_ids:
                model_label = get_format_data_from_model(base_model, [select_chessboards[id]])[0]
                model_label_mapped = apply_mapping_chess(model_label, abduced_map)
                #print(model_label_mapped, end=' ')
                model_labels.append(model_label)
                
            print("\nThis is the correct label of model:")
            for id in consistent_ex_ids:
                model_label_correct = get_format_data_from_model(correct_model, [select_chessboards[id]])[0]
                print(model_label_correct, end=' ')

            # Calcualte precision between model trained by abduced exs and abduced exs itself(before map)
            model_img_labels = np.array(get_chess_img_or_label(model_labels))
            assert(len(model_img_labels)==len(abduced_labels))
            batch_label_model_precision = ((model_img_labels==abduced_labels).sum()/len(model_img_labels))
            consistent_percentage = len(consistent_ex_ids)/len(select_chessboards)
            
            # Test if we can evaluate
            # The condition is: consistent_percentage>=0.8 && batch_label_model_precision>0.8
            '''
            print("Model's raw label")
            print(model_img_labels)
            print("Abduced raw labels")
            print(abduced_labels)
            print(batch_label_model_precision)
            '''
            print("\nConsistent percentage:", consistent_percentage)
            print("Batch label model precision:", batch_label_model_precision)
            if consistent_percentage>=0.9 and batch_label_model_precision>=0.9:
                condition_cnt += 1
            else:
                condition_cnt = 0
            
            
            #The condition has been satisfied continuously seven times
            if condition_cnt>=10: 
                '''
                #Generate several rules
                #Get training data and label and split it into train and evaluate
                #Train mlp
                #Evaluate and decide next course or restart

                out_rules = []
                for i in range(LOGIC_OUTPUT_DIM):
                    while True:
                        select_index = np.random.randint(len(chessboards_true), size=3)
                        select_chessboards = np.array(chessboards_true)[select_index]
                        _, __, ___, rule = get_abduced_chessboards_labels(base_model, select_chessboards, labels, [abduced_map], True, shape)
                        if rule != None:
                            break
                    out_rules.append(rule)
                print(out_rules)
            
                #Prepare MLP training data
                mlp_train_vectors, mlp_train_labels = get_mlp_data(chessboards_true, chessboards_false_by_len_train[course_num], base_model, out_rules, abduced_map)
                index = np.array(list(range(len(mlp_train_labels))))
                np.random.shuffle(index)
                mlp_train_vectors = mlp_train_vectors[index]
                mlp_train_labels = mlp_train_labels[index]
                
                
                #Train MLP
                mlp_model = NN_model.get_mlp_net(LOGIC_OUTPUT_DIM)
                mlp_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                mlp_model.fit(mlp_train_vectors, mlp_train_labels, epochs=60, batch_size=128)
                
                #Prepare MLP validation data
                mlp_val_vectors, mlp_val_labels = get_mlp_data(chessboards_true_by_len_validation[course_num], chessboards_false_by_len_validation[course_num], base_model, out_rules, abduced_map)
                '''
                
                # Validation
                print("**Now validation**")
                print("Abduced map:", abduced_map)
                #for mapping in gen_mappings([0,1,2], [0,1,2]):
                #    print(mapping)
                true_accuracy,false_accuracy,total_accuracy = evaluate_data(base_model, chessboards_true_val, chessboards_false_val, abduced_map) #mapping
                print("Validation accuracy:", true_accuracy, false_accuracy, total_accuracy)
                
                current_model_accuracy(base_model, './dataset/mnist_images', labels, shape, abduced_map)
                #input()
                if true_accuracy > 0.8 and false_accuracy > 0.8 and total_accuracy > 0.88: #Save model and go to next course
                    base_model.save_weights('%s_nlm_weights_%d.hdf5'%(src_data_name, course_num))
                    break
                else:
                    '''
                    # Restart current course: reload model
                    if course_num==0:
                        for i in range(len(base_model.layers)):
                            base_model.layers[i].set_weights(t_model.layers[i].get_weights())
                    else:
                        base_model.load_weights('%s_nlm_weights_%d.hdf5'%(src_data_name, course_num-1))
                    '''
                    print("Failed! Continue to train.")
                    condition_cnt = 0
            
    '''
    #Train final mlp model
    #Calcualte how many chessboards should be selected in each length
    select_equation_cnt = []
    if LOGIC_OUTPUT_DIM%EQUATION_LEN_CNT==0:
        select_equation_cnt = [LOGIC_OUTPUT_DIM//EQUATION_LEN_CNT]*EQUATION_LEN_CNT
    else:
        select_equation_cnt = [LOGIC_OUTPUT_DIM//EQUATION_LEN_CNT]*EQUATION_LEN_CNT
        select_equation_cnt[-1] += LOGIC_OUTPUT_DIM%EQUATION_LEN_CNT
    assert sum(select_equation_cnt) == LOGIC_OUTPUT_DIM
    
    #Abduce rules
    out_rules = []
    for chessboards_type in range(EQUATION_LEN_CNT):  #for each length of test chessboards
        for i in range(select_equation_cnt[chessboards_type]):   #for each length, there are select_equation_cnt[chessboards_type] rules
            while True:
                select_index = np.random.randint(len(chessboards_true_by_len_train[chessboards_type]), size=3)
                select_chessboards = np.array(chessboards_true_by_len_train[chessboards_type])[select_index]
                _, __, ___, rule = get_abduced_chessboards_labels(base_model, select_chessboards, labels, [abduced_map], True, (32, 32,3))
                if rule != None:
                    break
            out_rules.append(rule)
    print(out_rules)
    
    #Get mlp training data
    mlp_train_vectors = []
    mlp_train_labels = []
    for chessboards_type in range(EQUATION_LEN_CNT):  #for each length of test chessboards        
        mlp_train_len_vectors, mlp_train_len_labels = get_mlp_data(chessboards_true_by_len_train[chessboards_type], chessboards_false_by_len_train[chessboards_type], base_model, out_rules, abduced_map)
        if chessboards_type==0:
            mlp_train_vectors = mlp_train_len_vectors.copy()
            mlp_train_labels = mlp_train_len_labels.copy()
        else:
            mlp_train_vectors = np.concatenate((mlp_train_vectors,mlp_train_len_vectors),axis=0)
            mlp_train_labels = np.concatenate((mlp_train_labels,mlp_train_len_labels),axis=0)
    
    index = np.array(list(range(len(mlp_train_labels))))
    np.random.shuffle(index)
    mlp_train_vectors = mlp_train_vectors[index]
    mlp_train_labels = mlp_train_labels[index]
    
    for i in range(2):  #Try three times to find the best mlp 
        #Train MLP
        mlp_model = NN_model.get_mlp_net(LOGIC_OUTPUT_DIM)
        mlp_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        mlp_model.fit(mlp_train_vectors, mlp_train_labels, epochs=60, batch_size=128)
        
        #Test MLP
        for chessboards_type,(chessboards_true,chessboards_false) in enumerate(zip(chessboards_true_by_len_test, chessboards_false_by_len_test)):  #for each length of test chessboards        
            mlp_test_len_vectors, mlp_test_len_labels = get_mlp_data(chessboards_true, chessboards_false, base_model, out_rules, abduced_map)
            #print(len(mlp_test_len_vectors))
            #print(len(mlp_test_len_labels))
            #for index in range(len(mlp_test_len_vectors)):
            #    print(mlp_test_len_vectors[i])
            #    print(mlp_test_len_labels[i])
            #input()
            result = mlp_model.evaluate(mlp_test_len_vectors, mlp_test_len_labels, batch_size=128)
            print("The result of testing length %d chessboards is:"%(chessboards_type+5))
            print(result)
    '''
    # Testing
    print("***Start testing***")
    print("**Now test each course**")
    print("Abduced map:", abduced_map)
    for course_num in range(0, max(len(chessboards_true_by_len_test),len(chessboards_false_by_len_test)) ): #for each size of chessboards
        print("Course number:",course_num)
        if course_num>=len(chessboards_true_by_len_test):
            chessboards_true = []
        else:
            chessboards_true = chessboards_true_by_len_test[course_num]
        if course_num>=len(chessboards_false_by_len_test):
            chessboards_false = []
        else:
            chessboards_false = chessboards_false_by_len_test[course_num]
        true_accuracy,false_accuracy,total_accuracy = evaluate_data(base_model, chessboards_true, chessboards_false, abduced_map)
        
        print("Testing accuracy:", true_accuracy,false_accuracy,total_accuracy)
        
    print("**Now test the whole test dataset**")
    true_accuracy,false_accuracy,total_accuracy = evaluate_data(base_model, input_file_true_test, input_file_false_test, abduced_map)
    print("Testing accuracy:", true_accuracy,false_accuracy,total_accuracy)
    
    current_model_accuracy(base_model, './dataset/mnist_images', labels, shape, abduced_map)
    
    
    return base_model

if __name__ == "__main__":
    labels = ['0','1','2']
    src_dir = "./dataset"
    src_data_name = "mnist_images"
    input_shape = (28, 28, 1)
    src_path = os.path.join(src_dir,src_data_name)
    net_model_test(src_path = src_path, labels = labels, src_data_name = src_data_name, shape = input_shape) #just test net model
    net_model_pretrain(src_path = src_path, labels = labels, src_data_name = src_data_name, shape = input_shape)

    model = nlm_main_func(labels = labels, src_data_name = src_data_name, src_data_file = 'bin_chess_data_chessboard_size_3_8.pk', shape = input_shape)

    
