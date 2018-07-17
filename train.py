import sys
import json
import config
import subprocess
from re_instance_extractor import REInstanceExtractor
from mln_generator import MLNGenerator


def extract_re_instances(input_name):
    # input을 읽어서 관계를 추출할 instance들(문장/sbj-obj쌍/Feature) 목록을 생성한다.
    inst_extractor = REInstanceExtractor()
    file_name = config.data_path + input_name
    re_instance_list = inst_extractor.extract_re_instance_for_experiment(file_name)
    return re_instance_list

def write_markov_logic_network_data(re_instance_list, train_db_name):
    # instance 정보들을 Markov Logic Network에 들어가는 evidence grounding들로 만든다.
    MLNGenerator().write_mln_data_for_train(re_instance_list, train_db_name)

def run_alchemy_weight_learning(train_db_name):
    # Alchemy를 통해 Markov Logic Network Inference를 한다.
    bashCommand = "{} -d -i {} -o {} -t {} -ne Label,HasRel -dNumIter 15".format(config.alchemy_path+'learnwts',
                                                                                 config.data_path+'re.mln',
                                                                                 config.data_path + 're-learnt.mln',
                                                                                 config.data_path + train_db_name)
    result = subprocess.call(bashCommand.split())


def main():
    input_name = 'train_data' if len(sys.argv) < 2 else str(sys.argv[1])
    train_db_name = 'train.db' if len(sys.argv) < 4 else str(sys.argv[3])

    try:
        re_instance_list = extract_re_instances(input_name)
        write_markov_logic_network_data(re_instance_list, train_db_name)
        run_alchemy_weight_learning(train_db_name)
        print ('Alchemy : MLN weight learning finished')
        print ('Training Finished')
    except:
        print ("ERROR : " + str(sys.exc_info()[0]))

if __name__ == '__main__':
    main()

