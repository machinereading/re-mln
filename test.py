import sys
import json
import config
import subprocess
from re_instance_extractor import REInstanceExtractor
from mln_generator import MLNGenerator
from mln_result_extractor import MLNResultExtractor
from extration_ranker import ExtractRanker

def read_input(input_name):
    f = open(config.data_path+input_name, 'r', encoding='utf-8')
    data_obj_list = []
    for line in f:
        if (len(line) < 2):
            continue
        try:
            data = json.loads(line.strip())
            data_obj_list.append(data['sentence'][0])
        except:
            data_obj_list = []
    f.close()
    return data_obj_list


def extract_re_instances(input_name):
    # input을 읽어서 관계를 추출할 instance들(문장/sbj-obj쌍/Feature) 목록을 생성한다.
    inst_extractor = REInstanceExtractor()
    file_name = config.data_path + input_name
    re_instance_list = inst_extractor.extract_re_instance_for_experiment(file_name)
    return re_instance_list

def write_markov_logic_network_data(re_instance_list, test_db_name, ist_matching_name):
    # instance 정보들을 Markov Logic Network에 들어가는 evidence grounding들로 만든다.
    MLNGenerator().write_mln_data(re_instance_list, test_db_name, ist_matching_name)

def run_alchemy_inference(re_file_name,test_db_name):
    # Alchemy를 통해 Markov Logic Network Inference를 한다.
    bashCommand = "{} -ms -i {} -r {} -e {} -q Label,HasRel".format(config.alchemy_path+'infer',
                                                                    config.data_path+'re-learnt.mln',
                                                                    config.data_path+re_file_name,
                                                                    config.data_path+test_db_name)
    result = subprocess.call(bashCommand.split())


def get_spo_result_list(re_file_name, test_db_name, ist_matching_name):
    # MLN 결과 파일들로 부터 relation 목록(spo,relation,score)를 뽑아낸다.
    return MLNResultExtractor().get_re_result(re_file_name,test_db_name,ist_matching_name)

def write_output(spo_relation_result, output_name):
    # output 파일을 출력한다
    # sample : 애플_(기업)	foundedBy	스티브_워즈니악	.	0.992171806968	애플_(기업) 은 스티브_잡스 와 스티브_워즈니악 과 로널드_웨인 이 1976년에 설립한 컴퓨터 회사 이다.
    f = open(config.data_path+output_name,'w',encoding='utf-8')
    for result in spo_relation_result:
        f.write(result['sbj']+'\t'+result['relation']+'\t'+result['obj']+'\t'+'.'+'\t'+str(result['score'])+'\t'+result['sent']+'\n')
    f.close()

def main():
    input_name = 'test_data' if len(sys.argv) < 2 else str(sys.argv[1])
    output_name = 'result' if len(sys.argv) < 3 else str(sys.argv[2])
    refile_name = 're_test.result' if len(sys.argv) < 4 else str(sys.argv[3])
    test_db_name = 'test.db' if len(sys.argv) < 5 else str(sys.argv[4])
    ist_matching_name = 'instance_matching_test.txt' if len(sys.argv) < 6 else str(sys.argv[5])

    try:
        re_instance_list = extract_re_instances(input_name)
        write_markov_logic_network_data(re_instance_list, test_db_name, ist_matching_name)
        run_alchemy_inference(refile_name,test_db_name)
        print ('Alchemy : MLN inference finished')
        extract_ranker = ExtractRanker()
        extract_ranker.extract_rank()
        print('Test Finished')
    except:
        print ("ERROR : " + str(sys.exc_info()[0]))

if __name__ == '__main__':
    main()

