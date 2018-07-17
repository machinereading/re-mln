import numpy as np
import config
from collections import Counter

class MLNGenerator():
    def _get_entity_str(self,obj):
        return '(' + obj['sbj'] + '-@-' + obj['obj'] + ')'

    def _is_valid_morp(self,morp):
        morp_tag = morp.split('-@-')[1].strip()
        if (morp_tag.startswith('N') or morp_tag == 'VV' or morp_tag == 'VA'):
            return True
        return False

    def _get_dependency_feature_strs(self,obj):
        result1 = ''
        ary = obj['dependency_morp']
        for item in ary:
            if(self._is_valid_morp(item)):
                result1 += item + '--'
        if (len(result1) > 2):
            result1 = result1[:-2]

        ary = obj['dependency']
        result2 = ''
        for item in ary:
            if ('||' in item and ('down' in item or 'up' in item)):
                result2 += item + '--'
        if (len(result2) > 2):
            result2 = result2[:-2]
        return result1,result2

    def _get_feature_set_of_obj(self,obj):
        feature_set = set()
        lexical_dep, grammatic_dep = self._get_dependency_feature_strs(obj)
        if (len(lexical_dep) > 0):
            feature_set.add('#dependency_' + lexical_dep)
        if (len(grammatic_dep) > 0):
            feature_set.add('#dependency_' + grammatic_dep)
        for morp in obj['dependency_morp']:
            if (self._is_valid_morp(morp)):
                feature_set.add(morp)
        for morp in obj['morp_left']:
            if (self._is_valid_morp(morp)):
                feature_set.add(morp)
        for morp in obj['morp_right']:
            if (self._is_valid_morp(morp)):
                feature_set.add(morp)
        for morp in obj['morp_middle']:
            if (self._is_valid_morp(morp)):
                feature_set.add(morp)
        return  feature_set

    def write_mln_data_for_train(self, data, train_db_name):
        N = len(data)
        TRAIN_SIZE = len(data)+1

        entity_dict = {}
        argfea_dict = {}
        instance_dict = {}
        feature_counter = Counter()
        feature_dic = {}
        train_feature_vectors = []
        test_feature_vectors = []

        feature_count = 0
        entity_count = 0
        instance_count = 0
        relation_dict = {}
        ii = 0
        for obj in data:
            ii += 1
            entity_str = self._get_entity_str(obj)
            if (entity_str not in entity_dict):
                entity_count += 1
                entity_dict[entity_str] = 'P' + str(entity_count)

            instance_str = entity_str + '-@-' + obj['sent'].strip()
            instance_str += '___' + str(ii)

            if (instance_str not in instance_dict):
                instance_count += 1
                instance_dict[instance_str] = 'M' + str(instance_count)
            obj['instance_id'] = instance_dict[instance_str]

            if (ii < TRAIN_SIZE):
                feature_set_of_obj = self._get_feature_set_of_obj(obj)
                for feature in feature_set_of_obj:
                    feature_counter[feature] += 1
                if (obj['sbj_ne'] not in argfea_dict):
                    argfea_dict[obj['sbj_ne']] = obj['sbj_ne']
                if (obj['obj_ne'] not in argfea_dict):
                    argfea_dict[obj['obj_ne']] = obj['obj_ne']

                if (obj['relation'] not in relation_dict):
                    relation_dict[obj['relation']] = 0
                relation_dict[obj['relation']] += 1

        # 너무 많거나 적은 feature는 제거
        feature_max = int(N / 2)
        feature_min = 3
        feature_index = 0
        for feature in feature_counter:
            feature_min = 2 if ('#dependency_' in feature) else 5  # dependency path feature는 2번만 나타나도 ㅇㅋ
            if feature_counter[feature] >= feature_min and feature_counter[feature] <= feature_max:
                feature_index += 1
                feature_dic[feature] = 'F' + str(feature_index)

        #### Mutual Information 을 계산하기 위한 코드 ###
        # Mutual Information Mi(w) = log(pi(w)/Pi(w))
        # pi(w) = 주어진 클래스 k에서 word w의 frequency 비율/ Pi(w) = 전체 데이터에서 주어진 클래스 k의 비율

        all_word_freq = 0
        word_freq = 0
        rel_freq = {}
        rel_word_freq = {}
        for key in relation_dict:
            rel_freq[key] = 0

        for feature in feature_dic:
            rel_word_freq[feature] = {'__total__': 0}
            for key in relation_dict:
                rel_word_freq[feature][key] = 0

        i = 0
        for obj in data:
            i += 1
            if (i >= TRAIN_SIZE):
                break
            feature_set_of_obj = self._get_feature_set_of_obj(obj)
            rel = obj['relation']
            for feature in feature_set_of_obj:
                if feature in feature_dic:
                    rel_freq[rel] += 1
                    rel_word_freq[feature][rel] += 1
                    rel_word_freq[feature]['__total__'] += 1
                    all_word_freq += 1

        for rel in relation_dict:
            rel_freq[rel] = rel_freq[rel] / all_word_freq
            for feature in feature_dic:
                rel_word_freq[feature][rel] = rel_word_freq[feature][rel] / rel_word_freq[feature]['__total__']

        max_mutual = {}
        for feature in feature_dic:
            max_val = 0.0
            max_rel = ''
            for rel in relation_dict:
                if (rel_word_freq[feature][rel] == 0):
                    continue
                t = np.log(rel_word_freq[feature][rel] / rel_freq[rel])
                if (t > max_val):
                    max_val = t
                    max_rel = rel
            if (max_val < 0.0):
                debug = 1

            feature_id = feature_dic[feature]
            feature_id = int(feature_id[1:]) - 1
            max_mutual[feature_id] = max_val

        f = open(config.data_path+'feature_vector_weight.txt', 'w', encoding='utf-8')
        for feature in feature_dic:
            feature_id = feature_dic[feature]
            feature_id = int(feature_id[1:]) - 1
            f.write('%d\t%.4lf\n' % (feature_id, max_mutual[feature_id]))
        f.close()
        debug = 1

        #### Mutual Information 을 계산하기 위한 코드 끝 ###


        f_train = open(config.data_path+train_db_name, 'w', encoding='utf-8')
        f_test = open(config.data_path+'temp_test.db', 'w', encoding='utf-8')

        # Mention 출력
        index = 0
        hasrel_set = set()
        answer_hasrel_set = set()
        answer_set2 = set()
        mention_printed = [False for i in range(N + 1)]
        for obj in data:
            index += 1
            entity = self._get_entity_str(obj)
            instance_str = obj['instance_id']
            entity_str = entity_dict[entity]
            if (mention_printed[int(instance_str[1:])] == False):
                f_out = f_train if (index < TRAIN_SIZE) else f_test
                f_out.write('Mention(' + instance_str + ',' + entity_str + ')' + '\n')
                if (index < TRAIN_SIZE):
                    f_out.write('Label(' + instance_str + ',' + 'R_' + obj['relation'] + ')' + '\n')
                mention_printed[int(instance_str[1:])] = True

            if (index < TRAIN_SIZE):
                hasrel_set.add('HasRel(' + entity_str + ',' + 'R_' + obj['relation'] + ')')
            else:
                answer_hasrel_set.add(entity_str + '\t' + 'R_' + obj['relation'])
                answer_set2.add(instance_str + '\t' + 'R_' + obj['relation'])

        # Relation 출력
        for item in hasrel_set:
            f_train.write(item + '\n')

        # HasFea 출력
        index = 0
        is_printed = [False for i in range(N + 1)]
        num_features = len(feature_dic)
        for obj in data:
            index += 1
            instance_str = obj['instance_id']
            instance_id = int(instance_str[1:])
            if (is_printed[instance_id]):
                continue
            is_printed[instance_id] = True
            feature_set_of_obj = self._get_feature_set_of_obj(obj)
            idxs = []
            for feature in feature_set_of_obj:
                if feature in feature_dic:
                    feature_id = feature_dic[feature]
                    idxs.append(int(feature_id[1:]) - 1)
                    f_out = f_train if (index < TRAIN_SIZE) else f_test
                    f_out.write('HasFea(' + instance_str + ',' + feature_id + ')' + '\n')

            feature_vectors = train_feature_vectors if (index < TRAIN_SIZE) else test_feature_vectors
            feature_vectors.append((instance_str, sorted(idxs), obj['template_sent'].strip()))

        countt = 0

        for k in range(2):
            countt = 0
            done_cnt = 0
            feater_vector = train_feature_vectors if k is 0  else test_feature_vectors
            f_out = f_train if k is 0 else f_test
            for i in range(len(feater_vector)):
                instance1 = feater_vector[i]
                done_cnt += 1
                sim_list = []
                for j in range(i+1, len(feater_vector)):
                    instance2 = feater_vector[j]
                    if (instance1[0] == instance2[0]):
                        continue

                    vec1 = instance1[1]
                    vec2 = instance2[1]
                    idx1, idx2 = 0,0

                    size1=0.0
                    for idx in vec1:
                        size1 += (max_mutual[idx]*max_mutual[idx])
                    size1 = np.sqrt(size1)

                    size2=0.0
                    for idx in vec2:
                        size2 += (max_mutual[idx] * max_mutual[idx])
                    size2 = np.sqrt(size2)
                    size = 0.0
                    for idx1 in range(len(vec1)):
                        while idx2 < len(vec2) and vec1[idx1] >= vec2[idx2]:
                            if (vec1[idx1] == vec2[idx2]):
                                #size += 1
                                size += max_mutual[vec1[idx1]] * max_mutual[vec1[idx1]]
                            idx2 += 1
                        if (idx2>=len(vec2)):
                            break

                    cos_sim = size / (size1*size2) if (size1 > 0 and size2 > 0) else 0.0
                    if cos_sim >= 0.999 and instance1[2] == instance2[2]:
                        continue
                    if ( cos_sim >= 0.70 ):
                        sim_list.append((instance2[0],cos_sim))
                sim_list = sorted(sim_list, reverse=True, key=lambda tup: tup[1])
                t_count = 0
                for sim_item in sim_list:
                    f_out.write('Similar(' + instance1[0] + ',' + sim_item[0] + ')' + '\n')
                    t_count += 1
                    countt += 1
                    if (t_count >= 10):
                        break
                if (done_cnt % 50 == 0):
                    print('%d similarity calculation finished.' % (done_cnt))

        # Arg1HasFea, Arg2HasFea 출력
        index = 0
        is_printed = [False for i in range(N + 1)]
        for obj in data:
            index += 1
            instance_str = obj['instance_id']
            instance_id = int(instance_str[1:])
            if (is_printed[instance_id]):
                continue
            is_printed[instance_id] = True

            arg1type = obj['sbj_ne']
            arg2type = obj['obj_ne']
            f_out = f_train if (index < TRAIN_SIZE) else f_test
            if (arg1type != 'NONE' and arg1type in argfea_dict):
                f_out.write('Arg1HasFea(' + instance_str + ',' + arg1type + ')' + '\n')
            if (arg2type != 'NONE' and arg2type in argfea_dict):
                f_out.write('Arg2HasFea(' + instance_str + ',' + arg2type + ')' + '\n')

        f_train.close()
        f_test.close()

        # Entit-Pair Mapping
        f_write = open(config.data_path+'entity_pair_matching.txt', 'w', encoding='utf-8')
        for key in entity_dict:
            f_write.write(entity_dict[key] + '\t' + key + '\n')
        f_write.close()

        # Instance Mapping
        f_write = open(config.data_path+'instance_matching.txt', 'w', encoding='utf-8')
        for key in instance_dict:
            f_write.write(instance_dict[key] + '\t' + key + '\n')
        f_write.close()

        # Feature Mapping
        f_write = open(config.data_path+'feature_matching.txt', 'w', encoding='utf-8')
        for key in feature_dic:
            f_write.write(feature_dic[key] + '\t' + key + '\n')
        f_write.close()

        # Relation List
        f_write = open(config.data_path+'relation_list.txt', 'w', encoding='utf-8')
        for key in relation_dict:
            f_write.write('R_' + key + '\n')
        f_write.close()

    def write_mln_data(self,data,test_db_name,ist_matching_name):
        N = len(data)
        entity_dict = {}
        instance_dict = {}
        feature_dic = {}
        test_feature_vectors = []

        entity_count = 0
        instance_count = 0
        ii = 0
        for obj in data:
            ii += 1
            entity_str = self._get_entity_str(obj)
            if (entity_str not in entity_dict):
                entity_count += 1
                entity_dict[entity_str] = 'P_T' + str(entity_count)

            instance_str = entity_str + '-@-' + obj['sent'].strip()
            instance_str += '___' + str(ii)

            if (instance_str not in instance_dict):
                instance_count += 1
                instance_dict[instance_str] = 'M_T' + str(instance_count)
            obj['instance_id'] = instance_dict[instance_str]

        f = open(config.data_path+'feature_matching.txt','r',encoding='utf-8')
        for line in f:
            f_id, feature = line.split('\t')
            f_id = f_id.strip()
            feature = feature.strip()
            feature_dic[feature] = f_id
        f.close()

        #### Mutual Information 을 계산하기 위한 코드 ###
        # Mutual Information Mi(w) = log(pi(w)/Pi(w))
        # pi(w) = 주어진 클래스 k에서 word w의 frequency 비율/ Pi(w) = 전체 데이터에서 주어진 클래스 k의 비율
        max_mutual = {}
        f = open(config.data_path+'feature_vector_weight.txt','r',encoding='utf-8')
        for line in f:
            feature_id, value = line.split('\t')
            feature_id = int(feature_id.strip())
            value = float(value.strip())
            max_mutual[feature_id] = value
        f.close()

        f_test = open(config.data_path+test_db_name, 'w', encoding='utf-8')
        # Mention 출력
        index = 0
        hasrel_set = set()
        answer_hasrel_set = set()
        answer_set2 = set()
        mention_printed = [False for i in range(N + 1)]
        for obj in data:
            index += 1
            entity = self._get_entity_str(obj)
            instance_str = obj['instance_id']
            entity_str = entity_dict[entity]
            if (mention_printed[int(instance_str[3:])] == False):
                f_out = f_test
                f_out.write('Mention(' + instance_str + ',' + entity_str + ')' + '\n')
                mention_printed[int(instance_str[3:])] = True
            answer_hasrel_set.add(entity_str + '\t' + 'R_' + obj['relation'])
            answer_set2.add(instance_str + '\t' + 'R_' + obj['relation'])

        # HasFea 출력
        index = 0
        is_printed = [False for i in range(N + 1)]
        for obj in data:
            index += 1
            instance_str = obj['instance_id']
            instance_id = int(instance_str[3:])
            if (is_printed[instance_id]):
                continue
            is_printed[instance_id] = True
            feature_set_of_obj = self._get_feature_set_of_obj(obj)
            idxs = []
            for feature in feature_set_of_obj:
                if feature in feature_dic:
                    feature_id = feature_dic[feature]
                    idxs.append(int(feature_id[1:]) - 1)
                    f_out = f_test
                    f_out.write('HasFea(' + instance_str + ',' + feature_id + ')' + '\n')

            feature_vectors = test_feature_vectors
            feature_vectors.append((instance_str, sorted(idxs), obj['sent'].strip()))


        ############### Similar() 찍는 코드 시작 ########################################
        countt = 0
        done_cnt = 0
        feater_vector = test_feature_vectors
        f_out = f_test
        for i in range(len(feater_vector)):
            instance1 = feater_vector[i]
            done_cnt += 1
            sim_list = []
            for j in range(i+1, len(feater_vector)):
                instance2 = feater_vector[j]
                if (instance1[0] == instance2[0]):
                    continue

                vec1 = instance1[1]
                vec2 = instance2[1]
                idx1, idx2 = 0,0

                size1=0.0
                for idx in vec1:
                    size1 += (max_mutual[idx]*max_mutual[idx])
                size1 = np.sqrt(size1)

                size2=0.0
                for idx in vec2:
                    size2 += (max_mutual[idx] * max_mutual[idx])
                size2 = np.sqrt(size2)

                size = 0.0
                for idx1 in range(len(vec1)):
                    while idx2 < len(vec2) and vec1[idx1] >= vec2[idx2]:
                        if (vec1[idx1] == vec2[idx2]):
                            size += max_mutual[vec1[idx1]] * max_mutual[vec1[idx1]]
                        idx2 += 1
                    if (idx2>=len(vec2)):
                        break

                cos_sim = size / (size1*size2) if (size1 > 0 and size2 > 0) else 0.0
                if instance1[2] == instance2[2]:
                    continue
                if ( cos_sim >= 0.70):
                    sim_list.append((instance2[0],cos_sim))

            sim_list = sorted(sim_list, reverse=True, key=lambda tup: tup[1])
            t_count = 0
            for sim_item in sim_list:
                f_out.write('Similar(' + instance1[0] + ',' + sim_item[0] + ')' + '\n')
                t_count += 1
                countt += 1
                if (t_count >= 10):
                    break

        # Arg1HasFea, Arg2HasFea 출력
        index = 0
        is_printed = [False for i in range(N + 1)]
        for obj in data:
            index += 1
            instance_str = obj['instance_id']
            instance_id = int(instance_str[3:])
            if (is_printed[instance_id]):
                continue
            is_printed[instance_id] = True

            arg1type = obj['sbj_ne']
            arg2type = obj['obj_ne']
            f_out = f_test
            if (arg1type != 'NONE'):
                f_out.write('Arg1HasFea(' + instance_str + ',' + arg1type + ')' + '\n')
            if (arg2type != 'NONE'):
                f_out.write('Arg2HasFea(' + instance_str + ',' + arg2type + ')' + '\n')

        f_test.close()

        # Instance Mapping
        f_write = open(config.data_path+ist_matching_name, 'w', encoding='utf-8')
        for key in instance_dict:
            sbj,obj,sent = key.split('-@-')
            sbj = sbj[1:].strip()
            obj = obj[:-1].strip()
            sent = sent.strip().split('___')[0].strip()
            f_write.write(instance_dict[key] + '\t' + sbj + '\t' + obj + '\t' + sent + '\n')
        f_write.close()

        # Answer Set
        f_write = open(config.data_path + 'answer_set.txt', 'w', encoding='utf-8')
        for value in answer_hasrel_set:
            f_write.write(value + '\n')
        f_write.close()

        # Answer Set2
        f_write = open(config.data_path + 'answer_set2.txt', 'w', encoding='utf-8')
        for value in answer_set2:
            f_write.write(value + '\n')
        f_write.close()

    def write_mln_data_for_raw(self,data,test_db_name,ist_matching_name):
        N = len(data)
        entity_dict = {}
        instance_dict = {}
        feature_dic = {}
        test_feature_vectors = []

        entity_count = 0
        instance_count = 0
        ii = 0
        for obj in data:
            ii += 1
            entity_str = self._get_entity_str(obj)
            if (entity_str not in entity_dict):
                entity_count += 1
                entity_dict[entity_str] = 'P_T' + str(entity_count)

            instance_str = entity_str + '-@-' + obj['sent'].strip()
            instance_str += '___' + str(ii)

            if (instance_str not in instance_dict):
                instance_count += 1
                instance_dict[instance_str] = 'M_T' + str(instance_count)
            obj['instance_id'] = instance_dict[instance_str]

        f = open(config.data_path + 'pre_trained/feature_matching.txt', 'r', encoding='utf-8')
        for line in f:
            f_id, feature = line.split('\t')
            f_id = f_id.strip()
            feature = feature.strip()
            feature_dic[feature] = f_id
        f.close()

        #### Mutual Information 을 계산하기 위한 코드 ###
        # Mutual Information Mi(w) = log(pi(w)/Pi(w))
        # pi(w) = 주어진 클래스 k에서 word w의 frequency 비율/ Pi(w) = 전체 데이터에서 주어진 클래스 k의 비율
        max_mutual = {}
        f = open(config.data_path + 'pre_trained/feature_vector_weight.txt', 'r', encoding='utf-8')
        for line in f:
            feature_id, value = line.split('\t')
            feature_id = int(feature_id.strip())
            value = float(value.strip())
            max_mutual[feature_id] = value
        f.close()

        f_test = open(config.data_path + test_db_name, 'w', encoding='utf-8')
        # Mention 출력
        index = 0
        mention_printed = [False for i in range(N + 1)]
        for obj in data:
            index += 1
            entity = self._get_entity_str(obj)
            instance_str = obj['instance_id']
            entity_str = entity_dict[entity]
            if (mention_printed[int(instance_str[3:])] == False):
                f_out = f_test
                f_out.write('Mention(' + instance_str + ',' + entity_str + ')' + '\n')
                mention_printed[int(instance_str[3:])] = True

        # HasFea 출력
        index = 0
        is_printed = [False for i in range(N + 1)]
        for obj in data:
            index += 1
            instance_str = obj['instance_id']
            instance_id = int(instance_str[3:])
            if (is_printed[instance_id]):
                continue
            is_printed[instance_id] = True
            feature_set_of_obj = self._get_feature_set_of_obj(obj)
            idxs = []
            for feature in feature_set_of_obj:
                if feature in feature_dic:
                    feature_id = feature_dic[feature]
                    idxs.append(int(feature_id[1:]) - 1)
                    f_out = f_test
                    f_out.write('HasFea(' + instance_str + ',' + feature_id + ')' + '\n')

            feature_vectors = test_feature_vectors
            feature_vectors.append((instance_str, sorted(idxs), obj['sent'].strip()))

        ############### Similar() 찍는 코드 시작 ########################################
        countt = 0
        done_cnt = 0
        feater_vector = test_feature_vectors
        f_out = f_test
        for i in range(len(feater_vector)):
            instance1 = feater_vector[i]
            done_cnt += 1
            sim_list = []
            for j in range(i + 1, len(feater_vector)):
                instance2 = feater_vector[j]
                if (instance1[0] == instance2[0]):
                    continue

                vec1 = instance1[1]
                vec2 = instance2[1]
                idx1, idx2 = 0, 0

                size1 = 0.0
                for idx in vec1:
                    size1 += (max_mutual[idx] * max_mutual[idx])
                size1 = np.sqrt(size1)

                size2 = 0.0
                for idx in vec2:
                    size2 += (max_mutual[idx] * max_mutual[idx])
                size2 = np.sqrt(size2)

                size = 0.0
                for idx1 in range(len(vec1)):
                    while idx2 < len(vec2) and vec1[idx1] >= vec2[idx2]:
                        if (vec1[idx1] == vec2[idx2]):
                            size += max_mutual[vec1[idx1]] * max_mutual[vec1[idx1]]
                        idx2 += 1
                    if (idx2 >= len(vec2)):
                        break

                cos_sim = size / (size1 * size2) if (size1 > 0 and size2 > 0) else 0.0
                if instance1[2] == instance2[2]:
                    continue
                if (cos_sim >= 0.70):
                    sim_list.append((instance2[0], cos_sim))

            sim_list = sorted(sim_list, reverse=True, key=lambda tup: tup[1])
            t_count = 0
            for sim_item in sim_list:
                f_out.write('Similar(' + instance1[0] + ',' + sim_item[0] + ')' + '\n')
                t_count += 1
                countt += 1
                if (t_count >= 10):
                    break

        # Arg1HasFea, Arg2HasFea 출력
        index = 0
        is_printed = [False for i in range(N + 1)]
        for obj in data:
            index += 1
            instance_str = obj['instance_id']
            instance_id = int(instance_str[3:])
            if (is_printed[instance_id]):
                continue
            is_printed[instance_id] = True

            arg1type = obj['sbj_ne']
            arg2type = obj['obj_ne']
            f_out = f_test
            if (arg1type != 'NONE'):
                f_out.write('Arg1HasFea(' + instance_str + ',' + arg1type + ')' + '\n')
            if (arg2type != 'NONE'):
                f_out.write('Arg2HasFea(' + instance_str + ',' + arg2type + ')' + '\n')

        f_test.close()

        # Instance Mapping
        f_write = open(config.data_path + ist_matching_name, 'w', encoding='utf-8')
        for key in instance_dict:
            sbj, obj, sent = key.split('-@-')
            sbj = sbj[1:].strip()
            obj = obj[:-1].strip()
            sent = sent.strip().split('___')[0].strip()
            f_write.write(instance_dict[key] + '\t' + sbj + '\t' + obj + '\t' + sent + '\n')
        f_write.close()