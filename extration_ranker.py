import math
import config
from sklearn.metrics import f1_score,precision_score,recall_score

class ExtractRanker():
    def read_mln_db(self):
        '''
        DB 파일을 읽는다.
        '''
        instance_high_rel = {}
        instance_rels = {}

        f = open(config.data_path + 're_test.result','r',encoding='utf-8')
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

    def read_instance_mention(self):
        mentions = {}
        f = open(config.data_path + 'test.db', 'r', encoding='utf-8')
        for line in f:
            line = line.strip()
            if (len(line) < 1):
                continue
            if line.startswith('Mention'):
                #Mention(M724,P453)
                line = line.replace('Mention(','').replace(')','')
                mention, pair = line.split(',')
                if (pair not in mentions):
                    mentions[pair] = []
                mentions[pair].append(mention)
        f.close()
        return mentions

    def read_mln_result(self):
        '''
        probability가 설정된 mln result db 파일을 읽어와서
        HasRel과 Label 값을 읽어온다. 
        '''
        co_occur = {}
        fea_list = {}
        relation_list = []
        f = open(config.data_path + 'relation_list.txt','r',encoding='utf-8')
        for line in f:
            line = line.strip()
            if (len(line) > 1):
                relation_list.append(line)
                co_occur[line] = {}
                fea_list[line] = []

        f.close()

        feature_map = {}
        f = open(config.data_path + 'feature_matching.txt', 'r', encoding='utf-8')
        for line in f:
            line = line.strip()
            if (len(line) > 1):
                fea,fea_val = line.split('\t')
                fea = fea.strip()
                fea_val = fea_val.strip()
                feature_map[fea] = fea_val
        f.close()

        arg_fea_list = []
        f = open(config.data_path + 're-learnt.mln','r',encoding='utf-8')
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

        arg_fea_list = sorted(arg_fea_list, reverse=True, key=lambda tup: tup[1])
        for key in fea_list:
            fea_list[key] = sorted(fea_list[key], reverse=True, key=lambda tup: tup[1])
        debug = 1

        return co_occur, relation_list


    def calc_precision_recall(self,co_occur, relation_list, mentions, instance_rels, instance_high_rel, answer_set):
        rel_dic = {}
        index = 0
        true_val = []
        predict_val = []
        rel_cnt = {}
        rel_data = {}
        instance_dic = {}

        f = open(config.data_path + 'instance_matching_test.txt','r',encoding='utf-8')
        for line in f:
            if len(line) < 2:
                continue
            items = line.strip().split('\t')
            id = items[0].strip()
            sbj = items[1].strip()
            obj = items[2].strip()
            sent = items[3].strip()
            sent = sent.replace(sbj,' << _sbj_ >> ').replace(obj, ' << _obj_ >> ')
            instance_dic[id] = {'sbj':sbj, 'obj':obj, 'sent':sent}
        f.close()

        for relation in relation_list:
            rel_data[relation] = {'total' : 0, 'predict':0, 'right':0}
            rel_dic[relation] = index
            rel_cnt[relation] = 0
            index += 1

        f_write = open(config.data_path + 'prediction_result.txt', 'w', encoding='utf-8')
        index = 0
        notin_count = 0
        for answer in answer_set:
            instance, relation = answer.split('-@-')
            if (instance not in instance_high_rel):
                notin_count += 1
                continue
            instance = instance.strip()
            gold_rel = relation.strip()
            system_rel = instance_high_rel[instance][0]
            system_conf = instance_high_rel[instance][1]

            sbj = instance_dic[instance]['sbj']
            obj = instance_dic[instance]['obj']
            sent = instance_dic[instance]['sent']
            f_write.write('%s\t%s\t%s\t%s\t%.4f\t%s\n'%(sbj,obj,gold_rel[2:],system_rel[2:],system_conf,sent))

            true_val.append(rel_dic[gold_rel])
            rel_cnt[gold_rel] += 1
            predict_val.append(rel_dic[system_rel])

            rel_data[gold_rel]['total'] += 1
            if (system_rel == gold_rel):
                rel_data[gold_rel]['right'] += 1
            rel_data[system_rel]['predict'] += 1

            index += 1
        f_write.close()

        f_write = open(config.data_path + 'prec_recall_per_prop.txt', 'w', encoding='utf-8')
        total = 0
        accurate = 0
        for rel in rel_data:
            if (rel_data[rel]['total'] == 0):
                prec = 0.0
                recall = 0.0
            else:
                prec = (rel_data[rel]['right'] / rel_data[rel]['predict']) if rel_data[rel]['predict'] > 0 else 0.0
                recall = rel_data[rel]['right'] / rel_data[rel]['total']
                total += rel_data[rel]['total']
                accurate += rel_data[rel]['right']
            if ((prec + recall) < 0.0000001):
                f1 = 0.0
            else:
                f1 = 2*(prec*recall) / (prec+recall)
            f_write.write(
                '%s\t%.3f\t%.3f\t%.3f\t%d\n' % (rel[2:], prec, recall, f1, rel_cnt[rel]))
        f_write.write('%s\t%.3f\t%.3f\t%.3f\n' % ('average', (accurate/total), (accurate/total), (accurate/total)))
        f_write.close()

    def read_answer(self):
        answer_set = []
        f = open(config.data_path + 'answer_set2.txt', 'r', encoding='utf-8')
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue
            pair,relation = line.split('\t')
            answer_set.append(pair+'-@-'+relation)
        f.close()
        return answer_set


    def extract_rank(self):
        co_occur, relation_list = self.read_mln_result()
        mentions = self.read_instance_mention()
        instance_rels, instance_high_rel = self.read_mln_db()
        answer_set = self.read_answer()
        self.calc_precision_recall(co_occur, relation_list, mentions, instance_rels, instance_high_rel, answer_set)