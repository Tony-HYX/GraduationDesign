import os
import itertools
import random
import numpy as np
from PIL import Image
import pickle

def get_sign_path_list(data_dir, sign_names):
    sign_num = len(sign_names)
    index_dict = dict(zip(sign_names, list(range(sign_num))))
    ret = [[] for _ in range(sign_num)]
    for path in os.listdir(data_dir):
        if (path in sign_names):
            index = index_dict[path]
            sign_path = os.path.join(data_dir, path)
            for p in os.listdir(sign_path):
                ret[index].append(os.path.join(sign_path, p))
    return ret

def split_pool_by_rate(pools, rate, seed = None):
    if seed is not None:
        random.seed(seed)
    ret1 = []
    ret2 = []
    for pool in pools:
        random.shuffle(pool)
        num = int(len(pool) * rate)
        ret1.append(pool[:num])
        ret2.append(pool[num:])
    return ret1, ret2

def check_conflict(x, y, type, ret):
    for [x1,y1,type1] in ret:
        if x1==x and y1==y:
            return True
        if type==0 or type1==0:
            if x1==x or y1==y:
                return True
        if type==1 or type1==1:
            if x1-x==y1-y or x1-x==y-y1:
                return True
        if type==2 or type1==2:
            if x1==x or y1==y or x1-x==y1-y or x1-x==y-y1:
                return True
        if type<0 or type>2:
            print("Error type!!")
    return False
    
def seem_to_conflict(x, y, ret):
    for [x1,y1,type1] in ret:
        if x1==x or y1==y or x1-x==y1-y or x1-x==y-y1:
            return True
    return False

def generator_chess(chessboard_size, label):
    ret = []
    used_chess = []
    '''
    If fail, then generate chess that are like to conflict(x1==x or y1==y or x1-x==y1-y or x1-x==y-y1, but not the type)
    '''
    if label == 'positive': #Not conflict
        conflict_cnt = 0
        #Randomly select a chess and a type, test if it will conflict
        while True:
            x = random.randint(0,chessboard_size-1)
            y = random.randint(0,chessboard_size-1)
            if [x,y] in used_chess:
                continue
            if not seem_to_conflict(x,y,ret): # If not seem to conflict, then drop this chess with 80%
                if random.randint(0,9)>=2:
                    continue
            type = random.randint(0,2)
            if check_conflict(x,y,type,ret): 
                conflict_cnt += 1
            else:  #If no conflict, then add to ret
                used_chess.append([x,y])
                ret.append([x,y,type])
                conflict_cnt = 0
            if conflict_cnt >= 20: #If conflict, then retry, if conflict continuously ten times, over
                if len(used_chess) >= chessboard_size/2:
                    break;
                else:
                    ret = []
                    used_chess = []
                    conflict_cnt = 0
    elif label == 'negative': #Conflict only one time
        while True:
            x = random.randint(0,chessboard_size-1)
            y = random.randint(0,chessboard_size-1)
            if [x,y] in used_chess:
                continue
            type = random.randint(0,2)
            used_chess.append([x,y])
            if check_conflict(x,y,type,ret):  #If conflict, then add to ret and return
                ret.append([x,y,type])
                random.shuffle(ret)
                return ret
            else:  #If no conflict, then add to ret
                ret.append([x,y,type])

    return ret

def generator_chess_by_size(chessboard_size, require_num, label):
    ret = []
    while len(ret) < require_num:
        ret.append(generator_chess(chessboard_size, label))
    return ret

def generator_chess_by_max_size(min_chessboard_size, max_chessboard_size, num_per_size, label):
    ret = []
    for chessboard_size in range(min_chessboard_size, max_chessboard_size + 1):
        ret.extend(generator_chess_by_size(chessboard_size, require_num = num_per_size, label = label))
    return ret

def generator_chess_images(image_pools, chessboards, shape, seed, is_color):
    if (seed is not None):
        random.seed(seed)
    ret = []
    for chessboard in chessboards:
        data = []
        print(chessboard)
        for chess in chessboard:
            pass
            #print(chess)
            index = chess[2]
            pick = random.randint(0, len(image_pools[index]) - 1)
            if is_color:
                image = Image.open(image_pools[index][pick]).convert('RGB').resize(shape)
                image_array = np.array(image).reshape((shape[0],shape[1],3))
            else:
                image = Image.open(image_pools[index][pick]).convert('I').resize(shape)
                image_array = np.array(image).reshape((shape[0],shape[1],1))
            image_array = (image_array-127)*(1./128)
            data.append([chess[0],chess[1],image_array])
        ret.append(data)
    return ret

def generate_whole_chessboard(image_pools, chessboards, shape, label):
    h = shape[0]
    w = shape[1]
    chessboard_size = 8
    cnt = 0
    for chessboard in chessboards:
        chessboard_img = Image.new('L',(w*8,h*8),color="white")
        for chess in chessboard:
            index = chess[2]
            pick = random.randint(0, len(image_pools[index]) - 1)
            image = Image.open(image_pools[index][pick]).convert('L').resize(shape)
            chessboard_img.paste(image, box=(w*chess[0],h*chess[1],w*chess[0]+w,h*chess[1]+h))
        chessboard_img.save('dataset/chessboards/'+label+'/'+str(cnt)+'.png')
        cnt += 1
    

def get_chess_data(data_dir, sign_dir_lists, shape = (28, 28), min_chessboard_size = 2, max_chessboard_size = 10, tmp_file_prev = 
None, seed = None, train_num_per_size = 500, test_num_per_size = 100, is_color = False):
    tmp_file = ""
    if (tmp_file_prev is not None):
        tmp_file = "%s_chessboard_size_%d_%d.pk" % (tmp_file_prev, min_chessboard_size, max_chessboard_size)
    if (os.path.exists(tmp_file)):
        return pickle.load(open(tmp_file, "rb"))

    image_pools = get_sign_path_list(data_dir, sign_dir_lists)
    train_pool, test_pool = split_pool_by_rate(image_pools, 0.8, seed)

    ret = {}
    for label in ["positive", "negative"]:
        train_chess = generator_chess_by_max_size(min_chessboard_size, max_chessboard_size, num_per_size = train_num_per_size, label = label)
        test_chess = generator_chess_by_max_size(min_chessboard_size, max_chessboard_size, num_per_size = test_num_per_size, label = label)
        print(train_chess)
        print(test_chess)
        generate_whole_chessboard(train_pool, train_chess, shape, label)
        ret["train:%s" % (label)] = generator_chess_images(train_pool, train_chess, shape, seed, is_color)
        ret["test:%s" % (label)] = generator_chess_images(test_pool, test_chess, shape, seed, is_color)

    if (tmp_file_prev is not None):
        pickle.dump(ret, open(tmp_file, "wb"))
    return ret


'''
The easiest data:
Legal(Not conflict):
As many chess as possible
Stop when it cannot place any chess on the board
If a chess is line-diag, it can also be a line or a diag.

Illegal(Conflict):
Not too much chess...
Stop when the first conflict encountered
If a chess is a line or a diag, it can also be a line-diag.

You cannot distinguish a chessboard without label
'''

if __name__ == "__main__":
    data = get_chess_data(data_dir = "./dataset/mnist_images",\
                    sign_dir_lists = ['0', '1', '2'],\
                    shape = (28, 28),\
                    min_chessboard_size = 3,\
                    max_chessboard_size = 8,\
                    tmp_file_prev = "bin_chess_data",\
                    train_num_per_size = 500, \
                    test_num_per_size = 200, \
                    is_color = False
                    )
    
