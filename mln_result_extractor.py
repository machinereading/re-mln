import config

class MLNResultExtractor():
    def _read_mln_db(self,re_file_name):
        '''
        DB 파일을 읽는다.
        '''
        instance_high_rel = {}
        instance_rels = {}

        f = open(config.data_path+re_file_name,'r',encoding='utf-8')
        for line in f:
            line = line.strip()
            if (len(line) < 1):
                continue
            state, prob = line.split(' ')
            prob = float(prob)

            if (state.startswith('Label')):
                state = state.replace('Label(','').replace(')','')
                instance, relation = state.split(',')
                if (instance not in instance_rels):
                    instance_rels[instance] = {}
                if (instance not in instance_high_rel):
                    instance_high_rel[instance] = ('NULL',-1000000)

                instance_rels[instance][relation] = prob
                if (prob > instance_high_rel[instance][1]):
                    instance_high_rel[instance] = (relation,prob)

        f.close()
        return instance_rels, instance_high_rel

    def _read_instance_mention(self,test_db_name):
        mentions = {}
        f = open(config.data_path+test_db_name, 'r', encoding='utf-8')
        for line in f:
            line = line.strip()
            if (len(line) < 1):
                continue
            if line.startswith('Mention'):
                line = line.replace('Mention(','').replace(')','')
                mention, pair = line.split(',')
                if (pair not in mentions):
                    mentions[pair] = []
                mentions[pair].append(mention)
        f.close()
        return mentions


    def _read_mln_result(self):
        '''
        probability가 설정된 mln result db 파일을 읽어와서
        HasRel과 Label 값을 읽어온다. 
        '''
        co_occur = {}
        fea_list = {}
        relation_list = []
        f = open(config.data_path+'pre_trained/relation_list.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip()
            if (len(line) > 1):
                relation_list.append(line)
                co_occur[line] = {}
                fea_list[line] = []

        f.close()

        feature_map = {}
        f = open(config.data_path+'pre_trained/feature_matching.txt', 'r', encoding='utf-8')
        for line in f:
            line = line.strip()
            if (len(line) > 1):
                fea,fea_val = line.split('\t')
                fea = fea.strip()
                fea_val = fea_val.strip()
                feature_map[fea] = fea_val
        f.close()

        arg_fea_list = []
        f = open(config.data_path+'pre_trained/re-learnt.mln','r',encoding='utf-8')
        for line in f:
            line = line.strip()
            if (len(line) < 1 or line.startswith('//')):
                continue
            items = line.split('  ')
            if (len(items) is not 2):
                continue
            try:
                prob = float(items[0])
            except:
                continue

            logics = items[1].split(' v ')
            if (len(logics) is not 2):
                continue
            logics[0] = logics[0].strip()
            logics[1] = logics[1].strip()
            if (logics[0].startswith('!HasRel') and logics[1].startswith('!HasRel')):
                R1 = logics[0].replace(')','').replace('!HasRel(a1,','')
                R2 = logics[1].replace(')', '').replace('!HasRel(a1,', '')
                co_occur[R1][R2] = co_occur[R2][R1] = prob*-1
            else:
                if ('Arg' in items[1]):
                    arg_fea_list.append((items[1], prob))
                else:
                    logics = items[1].split(' v ')
                    rel = logics[0].replace(')','').replace('Label(a1,','').strip()
                    fea = logics[1].replace(')','').replace('!HasFea(a1,','').strip()
                    try:
                        fea_list[rel].append((fea+"_"+feature_map[fea],prob))
                    except:
                        debug = 1
        f.close()
        return co_occur, relation_list

    def _get_spo_list(self,instance_high_rel, ist_matching_name):
        f = open(config.data_path+ist_matching_name,'r',encoding='utf-8')
        instance_dic = {}
        for line in f:
            if(len(line) < 2):
                continue
            tt = line.strip().split('\t')
            if (instance_high_rel[tt[0]][1] >= config.threshold):
                instance_dic[tt[0]] = {
                    'sbj' : tt[1],
                    'obj' : tt[2],
                    'sent' : tt[3],
                    'relation' : instance_high_rel[tt[0]][0],
                    'score' : instance_high_rel[tt[0]][1]
                }

        result = []
        for i in range(len(instance_high_rel)):
            idx = i+1
            key_val = 'M_T' + str(idx)
            if (key_val in instance_dic):
                result.append(instance_dic[key_val])

        return result

    def get_re_result(self, re_file_name, test_db_name,ist_matching_name):
        co_occur, relation_list = self._read_mln_result()
        mentions = self._read_instance_mention(test_db_name)
        instance_rels, instance_high_rel = self._read_mln_db(re_file_name)
        return self._get_spo_list(instance_high_rel,ist_matching_name)


