import sys
import time
import os
import subprocess
from shutil import copyfile

def main():
    fidx = '1' if len(sys.argv) < 2 else str(sys.argv[1])
    fidx = str(fidx)

    st_time = int(time.time())

    file_name_list = []
    f = open('./data/sample/list'+fidx+'.txt','r',encoding='utf-8')
    for line in f:
        file_name_list.append(line.strip())
    f.close()

    in_name = 'input' + fidx
    out_name = 'output' + fidx
    re_name = 're_test' + fidx + ".result"
    test_db_name = 'test' + fidx + ".db"
    ist_matching_name = 'instance_matching_test' + fidx + ".txt"
    idx_cnt = 0
    done_cnt = 0
    FNULL = open(os.devnull, 'w')
    for fname in file_name_list:
        idx_cnt += 1
        elapsed_time = int(time.time() - st_time) / 60.0
        print( '%d processed %d done %.2f min elpased'%(idx_cnt-1, done_cnt, elapsed_time))

        origin_fname = './data/sample/dump_input/'+fname
        if os.path.isfile(origin_fname):
            copyfile(origin_fname, './data/'+in_name)
        else:
            continue
        print('%s copy input done : '%(fname) + str(idx_cnt))
        bashCommand = 'python3 extract_relation.py {} {} {} {} {}'.format(in_name,out_name,re_name,test_db_name,ist_matching_name)
        print (bashCommand)
        try:
            subprocess.call(bashCommand.split(), stdout=FNULL, stderr=subprocess.STDOUT)
        except:
            continue
        print ('python call done : '+ str(idx_cnt))

        origin_out_name = './data/' + out_name
        copyfile(origin_out_name, './data/sample/dump_output/' + fname)
        print('copy output done : ' + str(idx_cnt))
        done_cnt += 1

if __name__ == '__main__':
    main()

