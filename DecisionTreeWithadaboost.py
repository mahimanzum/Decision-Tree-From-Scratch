import numpy as np
import pandas as pd
from numpy import log2 as log
from sklearn.preprocessing import Imputer

eps = np.finfo(float).eps

from sklearn.utils import shuffle

# print(eps)
dataset = {'Taste': ['Salty', 'Spicy', 'Spicy', 'Spicy', 'Spicy', 'Sweet', 'Salty', 'Sweet', 'Spicy', 'Salty'],
           'Temperature': ['Hot', 'Hot', 'Hot', 'Cold', 'Hot', 'Cold', 'Cold', 'Hot', 'Cold', 'Hot'],
           'Texture': ['Soft', 'Soft', 'Hard', 'Hard', 'Hard', 'Soft', 'Soft', 'Soft', 'Soft', 'Hard'],
           'Eat': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes']}

df = pd.DataFrame(dataset, columns=list(dataset.keys()))


def preProcessTelco():
    # https://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Preprocessing-Missing-Data-Categorical-Data.php
    # imputer = Imputer(missing_values ='NaN', strategy = 'mean', axis = 0)
    df = pd.read_csv('telco.csv')
    df.dropna(subset=['Churn'])
    df.drop('customerID', axis=1, inplace=True)
    df.drop('TotalCharges', axis=1, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
    df.loc[:, 'SeniorCitizen'].replace([1, 0], ["Yes", "No"], inplace=True)
    print(df.dtypes)
    # imputer = imputer.fit(df)
    # imputer_data = Imputer.fit_transform(df['tenure'].values)
    for i in range(len(df.keys())):
        if (df.dtypes[i] == 'int64' or df.dtypes[i] == 'float64'):
            mean = df[df.keys()[i]].mean()
            df[df.keys()[i]] = df[df.keys()[i]].fillna(mean)
            print(df.keys()[i])
    # if still there is missing data than is is catagorical so drop that row
    st = []
    for i in range(len(df.keys())):
        st.append(df.keys()[i])
    df.dropna(subset=st)

    pos_df = df[df['Churn'] == 'Yes']
    neg_df = df[df['Churn'] == 'No']
    min_length = min(len(pos_df), len(neg_df))
    print(len(pos_df), len(neg_df))

    # for same pos neg
    # test_df = pd.concat([pos_df[-(int)(0.2*min_length):],neg_df[-(int)(0.2*min_length):]])
    # test_df = test_df.reset_index(drop=True)
    # df = pd.concat([pos_df[:(int)(0.8*min_length)],neg_df[0:(int)(0.8*min_length)]])
    # df = df.reset_index(drop=True)

    test_df = pd.concat([pos_df[-(int)(0.2 * len(pos_df)):], neg_df[-(int)(0.2 * len(neg_df)):]])
    test_df = test_df.reset_index(drop=True)
    df = pd.concat([pos_df[:(int)(0.8 * len(pos_df))], neg_df[0:(int)(0.8 * len(neg_df))]])
    df = df.reset_index(drop=True)

    # df = shuffle(df)
    return shuffle(df), shuffle(test_df)


def preProcessCreditcard():
    # https://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Preprocessing-Missing-Data-Categorical-Data.php
    # imputer = Imputer(missing_values ='NaN', strategy = 'mean', axis = 0)
    df = pd.read_csv('creditcard.csv')
    # df = pd.DataFrame(dataset,columns= list(dataset.keys()))

    df['Class'] = df['Class'].astype(object)
    df.dropna(subset=['Class'])
    # print(df.dtypes)
    df.drop('Time', axis=1, inplace=True)
    # df['Class'] = df['Class'].map({'1':"Yes", '0':"No"})
    df.loc[:, 'Class'].replace([1, 0], ["Yes", "No"], inplace=True)
    print(df.dtypes)
    # print(df.head)
    for i in range(len(df.keys())):
        if (df.dtypes[i] == 'int64' or df.dtypes[i] == 'float64'):
            mean = df[df.keys()[i]].mean()
            df[df.keys()[i]] = df[df.keys()[i]].fillna(mean)
            print(df.keys()[i])

    # if still there is missing data than is is catagorical so drop that row
    st = []
    for i in range(len(df.keys())):
        st.append(df.keys()[i])
    df.dropna(subset=st)

    pos_df = df[df['Class'] == 'Yes'].reset_index(drop=True)
    neg_df = df[df['Class'] == 'No'].reset_index(drop=True)
    min_length = min(len(pos_df), len(neg_df))
    print(len(pos_df), len(neg_df))
    # test_df = pd.concat([pos_df[-(int)(0.2*min_length):],neg_df[-(int)(0.2*min_length):]])
    # test_df = test_df.reset_index(drop=True)

    df = pd.concat([pos_df, neg_df[:20000]]).reset_index(drop=True)
    df = shuffle(df).reset_index(drop=True)

    test_df = df[-(int)(0.2 * len(df)):].reset_index(drop=True)
    df = df[:(int)(0.8 * len(df))].reset_index(drop=True)
    print(len(df))
    print(len(test_df))

    test_df = pd.concat([pos_df[-(int)(0.2 * min_length):], neg_df[-(int)(0.2 * min_length):]]).reset_index(drop=True)
    df = pd.concat([pos_df[:(int)(0.8 * min_length)], neg_df[:(int)(0.8 * min_length)]]).reset_index(drop=True)

    return shuffle(df), shuffle(test_df)


def preProcessAdult():
    cols = []
    for i in range(14):
        cols.append("feature" + str(i))
    cols.append("Class")
    df = pd.read_csv('adult_train.csv', names=cols, header=None)
    test_df = pd.read_csv('adult_test.csv', names=cols, header=None)
    test_df = test_df[1:]
    test_len = len(test_df)
    df = pd.concat([df, test_df]).reset_index(drop=True)
    # remove all rows with question marks
    df = df[(df != '?').all(axis=1)]
    df = df[(df != ' ?').all(axis=1)]
    df = df[(df != np.nan).all(axis=1)]
    # test_df = test_df[(test_df != '?').all(axis=1)]

    df.dropna(subset=['Class'])
    df['Class'] = df['Class'].astype(object)
    # df.feature0 = df.feature0.astype(float).fillna(0.0)
    # df['feature0'] = df['feature0'].astype(int)

    # to convert feature 0
    frm = []
    to = []
    for i in range(100):
        frm.append(str(i))
        to.append(i)
    frm.append('|1x3 Cross validator')
    to.append(0)
    df = df[(df != '?').all(axis=1)]
    df = df[(df != ' ?').all(axis=1)]
    df = df[(df != np.nan).all(axis=1)]

    # df.loc[:, 'feature0'].replace(frm,to, inplace=True)
    df.dropna(subset=cols)

    # mean = df['feature0'].mean()
    # mean = float(int(mean))
    df['feature0'] = df['feature0'].astype(float)  # .fillna(mean)
    mean = df['feature0'].mean()
    df['feature0'] = df['feature0'].astype(float)  # .fillna(mean)
    print(to)
    print("start")
    print(df['feature0'].unique())
    print("end")
    # print(df.head)
    # print(df['Class'].unique())
    df.loc[:, 'Class'].replace([" <=50K", " >50K"], ["Yes", "No"], inplace=True)
    df.loc[:, 'Class'].replace([" <=50K.", " >50K."], ["Yes", "No"], inplace=True)
    # print(df.dtypes)
    # print(df.head)
    for i in range(len(df.keys())):
        if (df.dtypes[i] == 'int64' or df.dtypes[i] == 'float64'):
            mean = df[df.keys()[i]].mean()
            df[df.keys()[i]] = df[df.keys()[i]].fillna(mean)
            # print(df.keys()[i])

    # if still there is missing data than is is catagorical so drop that row
    st = []
    for i in range(len(df.keys())):
        st.append(df.keys()[i])
    df.dropna(subset=st)
    test_len = (int)(0.2 * len(df))

    test_df = df[-test_len:].reset_index(drop=True)
    df = df[:-test_len].reset_index(drop=True)
    print(df.dtypes)

    return shuffle(df), shuffle(test_df)

def preProcessOnline():
    df = pd.read_csv("online1_data.csv")
    df.loc[:, 'Class'].replace([1, 0], ["Yes", "No"], inplace=True)
    print(df.head())
    df = df[(df != np.nan).all(axis=1)]
    shuffle(df)
    pos_df = df[df['Class'] == 'Yes'].reset_index(drop=True)
    neg_df = df[df['Class'] == 'No'].reset_index(drop=True)
    print(df.dtypes)
    test_df = pd.concat([pos_df[-(int)(0.2 * len(pos_df)):], neg_df[-(int)(0.2 * len(pos_df)):]]).reset_index(drop=True)
    df = pd.concat([pos_df[:(int)(0.8 * len(neg_df))], neg_df[:(int)(0.8 *len(neg_df))]]).reset_index(drop=True)

    return shuffle(df), shuffle(test_df)


#df, test_df = preProcessTelco()
df, test_df = preProcessCreditcard()
#df, test_df = preProcessAdult()
#df, test_df = preProcessOnline()
# print(df.dtypes)

used = {}
uniqueFeaturesLables = {}
for attr in list(df.keys()):
    uniqueFeaturesLables[attr] = []
# print(df['Taste'])


for attr in list(df.keys()):
    for val in df[attr]:
        if (val not in uniqueFeaturesLables[attr]):
            uniqueFeaturesLables[attr].append(val)
'''
for attr in list(df.keys()):
    print(uniqueFeaturesLables[attr])
print(len(uniqueFeaturesLables))
'''

# df = pd.DataFrame(dataset)
# print(df.dtypes)
# print(df)
# print(test_df.dtypes)
print(len(test_df))
print(len(df))
# test_df[feature3].value_counts()[nan]

for i in test_df.keys():
    print(i)
    print(test_df[i].unique())



# test_df.head
# test 196
# train 786
# print(df[:400])
# print((df.iloc[10]['Class']))
# print(len(df.iloc[10]['Class']))

# print((test_df.iloc[10]['Class']))
# print(len(test_df.iloc[10]['Class']))
def Entropy(df, attr=None):
    label = df.keys()[-1]
    values = df[label].unique()
    # print(values)
    entropy = 0.0

    for value in values:
        fraction = df[label].value_counts()[value] / len(df[label])
        entropy += -fraction * np.log2(fraction)
    if attr == None:
        return (entropy)
    entropy = 0.0
    final_entropy = 0
    attr_variables = df[attr].unique()
    for variable in attr_variables:
        entropy = 0.0
        for value in values:
            num = len(df[attr][df[attr] == variable][df[label] == value])
            den = len(df[attr][df[attr] == variable])
            fraction = num / (den + eps)
            # print(den)
            entropy += -fraction * log(fraction + eps)
            fraction2 = den / len(df)
            final_entropy += fraction2 * entropy
    # print(attr_variables)
    return (final_entropy)


def ContEntropy(df, attr):
    label = df.keys()[-1]
    values = df[label].unique()
    entropy = 0.0
    total_yes = 0
    total_no = 0
    now_yes = 0
    now_no = 0
    total = 0
    for value in values:
        fraction = df[label].value_counts()[value] / len(df[label])
        if value == 'Yes':
            total_yes = df[label].value_counts()[value]
        else:
            total_no = df[label].value_counts()[value]
        entropy += -fraction * np.log2(fraction)
    total = total_yes + total_no
    entropy = 0.0
    final_entropy = 100
    partition_value = 0
    # attr_variables = df[attr].unique()
    attr_variables = []
    for i in range(len(df)):
        # print(df[attr][i])
        # print(i)
        attr_variables.append((df[attr][i], df[label][i]))
    attr_variables = sorted(attr_variables, key=lambda element: element[0])
    # print((attr_variables))

    for i in range(len(attr_variables)):
        entropy = 0.0
        if (attr_variables[i][1] == 'Yes'):
            now_yes += 1
        else:
            now_no += 1
        other_yes = total_yes - now_yes
        other_no = total_no - now_no
        if (i + 1 != len(attr_variables) and attr_variables[i][0] == attr_variables[i + 1][0]):
            continue
        entropy += -((now_yes + now_no) / (total + eps)) * (
                ((now_yes / (now_yes + now_no + eps)) * log(now_yes / (now_yes + now_no + eps))) + (
                (now_no / (now_yes + now_no + eps)) * log(now_no / (now_yes + now_no + eps))))
        entropy += -((other_yes + other_no) / (total + eps)) * (
                ((other_yes / (other_yes + other_no + eps)) * log(other_yes / (other_yes + other_no + eps))) + (
                (other_no / (other_yes + other_no + eps)) * log(other_no / (other_yes + other_no + eps))))
        if (entropy <= final_entropy):
            partition_value = attr_variables[i][0]
        final_entropy = min(final_entropy, entropy)
    '''
    for variable in attr_variables:
        entropy = 0.0
        for value in values:
            num = len(df[attr][df[attr] <= variable][df[label] ==value])
            den = len(df[attr][df[attr] <= variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        if(entropy <= final_entropy):
            partition_value = variable
        final_entropy = min(final_entropy, entropy)
    '''
    return final_entropy, partition_value


def find_winner(df, used):
    # print(df.keys())
    # Entropy_att = []
    gain = []
    chose = []
    partition = []
    # used = list(used)
    for key in df.keys()[:-1]:
        # print(key)
        # print(df.dtypes[key])
        if key not in used:
            chose.append(key)
            if df.dtypes[(key)] == 'object':
                # print("dhukse")
                gain.append(Entropy(df) - Entropy(df, key))
                partition.append("Null")
            else:
                # print("dhuke nai")
                # print("ashse" + str(key))
                entrpy, parttn = ContEntropy(df, key)
                gain.append(Entropy(df) - entrpy)
                partition.append(parttn)
    gain = np.array(gain)
    idx = np.argmax(gain)
    # print(partition)
    return chose[idx], partition[idx]


def get_subtable(df, index, value):
    return df[df[index] == value].reset_index(drop=True)


def get_subtable_cont(df, index, value):
    less_table = df[df[index] <= value].reset_index(drop=True)
    greater_table = df[df[index] > value].reset_index(drop=True)
    return less_table, greater_table


#find_winner(df, [])


# ContEntropy(df, 'tenure')


def makeTree(df, depth, max_depth, used, tree=None):
    # print("used = ")
    # print(used)
    Class = df.keys()[-1]
    # print(Class)
    # if(len(df[Class].unique())==1):
    #    return str(df[Class].unique()[0])
    # print(Class)

    values = df[Class].unique()
    x = 0
    majority_class = values[0]
    for value in values:
        # print(value)
        y = df[Class].value_counts()[value]
        if (y > x):
            x = y
            majority_class = value
    if depth == len(uniqueFeaturesLables) - 1 or depth == max_depth:
        return str(majority_class)

    partition_index, cont_partition_val = find_winner(df, used)

    # print(partition_index)
    # partition_values= np.unique(df[partition_index])
    partition_values = uniqueFeaturesLables[partition_index]
    if tree == None:
        tree = {}
        tree[partition_index] = {}
    # values = uniqueFeaturesLables[Class]

    if cont_partition_val == "Null":

        for value in partition_values:
            table = get_subtable(df, partition_index, value)
            ClassVal, cnt = np.unique(table[Class], return_counts=True)
            # print("ClassVal = "+ ClassVal);
            if len(cnt) == 1:
                tree[partition_index][value] = ClassVal[0]
            elif (len(table) == 0):
                tree[partition_index][value] = str(majority_class)
            else:
                used.append(partition_index)
                tree[partition_index][value] = makeTree(table, depth + 1, max_depth, used)
                used.pop()
    else:
        less_table, greater_table = get_subtable_cont(df, partition_index, cont_partition_val)
        less_ClassVal, less_cnt = np.unique(less_table[Class], return_counts=True)
        greater_ClassVal, greater_cnt = np.unique(less_table[Class], return_counts=True)
        done = 0
        # print("ClassVal = "+ ClassVal);
        if (done == 0):
            if len(less_cnt) == 1:
                tree[partition_index]["<=" + str(cont_partition_val)] = less_ClassVal[0]
            elif (len(less_table) == 0):
                tree[partition_index]["<=" + str(cont_partition_val)] = str(majority_class)
            else:
                print(used)
                print(partition_index)
                used.append(partition_index)
                tree[partition_index]["<=" + str(cont_partition_val)] = makeTree(less_table, depth + 1, max_depth, used)
                used.pop()
        if (done == 0):
            if len(greater_cnt) == 1:
                tree[partition_index][str(cont_partition_val)] = greater_ClassVal[0]
            elif (len(greater_table) == 0):
                tree[partition_index][str(cont_partition_val)] = str(majority_class)
            else:
                used.append(partition_index)
                tree[partition_index][str(cont_partition_val)] = makeTree(greater_table, depth + 1, max_depth, used)
                used.pop()

    return tree


tree = makeTree(df, 0, 200, [])
import pprint

# tree = makeTree(df, depth = 0, max_depth = 2)
#pprint.pprint(tree)

import pprint

# tree = makeTree(df, depth = 0, max_depth = 2)
#pprint.pprint(tree)

unique_labels = df[df.keys()[-1]].unique()
unique_labels
print(unique_labels)


def predict(tree, inst):
    # pprint.pprint(tree)
    for nodes in tree.keys():
        if (nodes in unique_labels):
            return nodes
        value = inst[nodes]

        if df.dtypes[nodes] != 'object':
            # print(list(tree[nodes].keys())[0])

            if list(tree[nodes].keys())[0][0] == '<':
                less_value = list(tree[nodes].keys())[0]
                greater_value = list(tree[nodes].keys())[1]
            else:
                less_value = list(tree[nodes].keys())[1]
                greater_value = list(tree[nodes].keys())[0]

            if ((df.dtypes[nodes] == 'int64' and inst[nodes] <= int(less_value[2:])) or (
                    df.dtypes[nodes] == 'float64' and inst[nodes] <= float(less_value[2:]))):
                tree = (tree[nodes][less_value])
            else:
                tree = (tree[nodes][greater_value])
        else:
            tree = tree[nodes][value]
        # print("nodes = " + nodes)
        # print("value = " + value)
        # tree = (tree[nodes][value])

        prediction = 'No'
        if type(tree) is dict:
            prediction = predict(tree, inst)
        else:
            prediction = tree
            break;
    return prediction
    if prediction == 'Yes':
        return 1.0
    return -1.0


inst = df.iloc[6]
print(inst)
# print(inst['Taste'])
print(predict(tree, inst))
# for i in range(10):
# print(predict(tree, df.iloc[i]))
# for i in range(10):
# print(predict(tree, df.iloc[i]))

# only to run ada-boost

# returns a vector of trees and weights of trees
from numpy.random import choice

feature_list = df.keys()

'''
def make_feedable(data):
    data_dic = {}
    for i in feature_list:
        data_dic[i] = []
    for j in feature_list:
        for i in data:
            data_dic[j].append(df[j][i])
    # print(data_dic)
    dic_df = pd.DataFrame(data_dic, columns=feature_list)
    return dic_df


def AdaBoost(df, K=5):
    data_weights = []
    Hypothesis_vector = []
    Hypothesis_weights = []
    list_of_candidates = []
    sampled = []
    for i in range(len(df)):
        data_weights.append(1.0 / len(df))
        sampled.append(i)
    data_weights = np.array(data_weights)
    # print(data_weights[i])
    # print(data_weights.values())

    for j in range(len(df)):
        list_of_candidates.append(j)

    i = 0
    # change it
    new_df = df.copy()
    while (i < K):
        if i != 0:
            sampled = choice(list_of_candidates, int(len(df)), p=list(data_weights))
            new_df = make_feedable(sampled)
        tree = makeTree(new_df, 0, 1, [])

        pprint.pprint(tree)
        # print(sampled)
        print(np.sum(data_weights))
        error = 0.0
        for j in range(len(df)):
            prediction = predict(tree, df.iloc[j])
            if (prediction != df.iloc[j][df.keys()[-1]]):
                error = error + data_weights[j]
        print(error)
        if error > 0.5:
            # print(i)
            continue
        for j in range(len(new_df)):
            prediction = predict(tree, new_df.iloc[sampled[j]])
            if prediction == new_df.iloc[sampled[j]][new_df.keys()[-1]]:
                data_weights[sampled[j]] = (data_weights[sampled[j]] * error) / (1 - error + eps)
        # normalization_step
        print(np.sum(data_weights))
        data_weights /= np.sum(data_weights)

        Hypothesis_weights.append(log((1 - error) / error))
        Hypothesis_vector.append(tree)
        pprint.pprint(tree)
        print("at  " + str(i) + "  accuracy = " + str(1 - error))
        # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice

        # sampled = choice(list_of_candidates, int(len(df)), p=list(data_weights))
        i = i + 1

    return Hypothesis_vector, Hypothesis_weights


Hypothesis_vector, Hypothesis_weights = AdaBoost(df, 20)


# only after running adaboost

def final_prediction(Hypothesis_vector, Hypothesis_weights, instance):
    outpt = 0.0
    for i in range(len(Hypothesis_vector) - 15):
        if predict(Hypothesis_vector[i], instance) == 'Yes':
            outpt = outpt + Hypothesis_weights[i]
        else:
            outpt = outpt - Hypothesis_weights[i]
    if outpt >= 0:
        return "Yes"
    else:
        return "No"


# for i in range(len(Hypothesis_vector)):
# print(final_prediction(Hypothesis_vector, Hypothesis_weights, df.iloc[i]))
#    pprint.pprint(Hypothesis_vector[i])
'''
def accuracy(df):
    correct = 0
    for i in range(len(df)):
        # pred = final_prediction(Hypothesis_vector, Hypothesis_weights, df.iloc[i])
        pred = predict(tree, df.iloc[i])
        if df.iloc[i][df.keys()[-1]] == pred:
            correct = correct + 1
    return correct / len(df)


print(len(df))


# print(accuracy(df))
# print(accuracy(test_df))

# set 1 0.7265415549597856
# set 2 0.8928571428571429
# set 3 0.8030340253040167

def calc_all(df):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i in range(len(df)):
        if (i % 100 == 0):
            print(i)
        # pred = final_prediction(Hypothesis_vector, Hypothesis_weights, df.iloc[i])
        pred = predict(tree, df.iloc[i])
        if df.iloc[i][df.keys()[-1]] == "Yes" and pred == "Yes":
            true_pos += 1
        if df.iloc[i][df.keys()[-1]] == "Yes" and pred == "No":
            false_neg += 1
        if df.iloc[i][df.keys()[-1]] == "No" and pred == "Yes":
            false_pos += 1
        if df.iloc[i][df.keys()[-1]] == "No" and pred == "No":
            true_neg += 1
    return true_pos, false_pos, true_neg, false_neg


true_pos, false_pos, true_neg, false_neg = calc_all(test_df)

tp_w = 1
tn_w = 1
fp_w = 1
fn_w = 1

true_pos *=tp_w
false_pos *= fp_w
true_neg *= tn_w
false_neg *= fn_w


accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
recall_val = true_pos / (true_pos + false_neg + eps)
precision_val = true_pos / (true_pos + false_pos + eps)
f1_score = 2 * precision_val * recall_val / (eps + precision_val + recall_val)
false_positive = false_pos / (true_neg + false_pos + eps)
fdr = false_pos / (true_pos + false_pos + eps)
print("accuracy = ", accuracy)
print("True positive rate /sensitivity, recall, hit rate = ", recall_val)
print("True negative rate specificity", 1 - recall_val)
print("precision = ", precision_val)
print("f1_score = ", f1_score)
print("False Discovery Rate", fdr)
print("false_positive = ", false_positive)

'''
k = 20 adult dataset
training 0.79412902869147
testing 0.7953339230429014

k = 15 adult dataset
training 0.79412902869147
testing 0.7953339230429014

k = 10 adult dataset
0.79412902869147
0.7953339230429014

k = 5 adult dataset
0.7514511581624191
0.7549756744803184

'''

'''
k = 20 credit card
0.9905447447081072
0.9895070766227428

k = 15 credit card
0.9903617397669737
0.9895070766227428

k = 10 credit card
0.9884706887085951
0.9877989263055149

k = 5 credit card
0.9883486854145062
0.9877989263055149
'''

'''
k = 20 telco
0.7346467873624423
0.7348969438521677

k = 15 telco
0.7346467873624423
0.7348969438521677

k = 10 telco
0.7346467873624423
0.7348969438521677

k = 5 telco
0.7346467873624423
0.7348969438521677


'''
'''
techno train dataset with full depth
accuracy =  0.939297124600639
True positive rate /sensitivity, recall, hit rate =  0.888963210702
True negative rate specificity 0.111036789298
precision =  0.883056478405
f1_score =  0.886
False Discovery Rate 0.116943521595
false_positive =  0.0425223483933

techno test dataset with full depth
accuracy =  0.7348969438521677
True positive rate /sensitivity, recall, hit rate =  0.498659517426
True negative rate specificity 0.501340482574
precision =  0.5
f1_score =  0.49932885906
False Discovery Rate 0.5
false_positive =  0.179883945841

creditcard train dataset with full depth with same pos and neg
accuracy =  0.8944386090994527
True positive rate /sensitivity, recall, hit rate =  0.947583314942
True negative rate specificity 0.0524166850585
precision =  0.914970697922
f1_score =  0.930991489131
False Discovery Rate 0.0850293020778
false_positive =  0.266236654804

creditcard test dataset with full depth with same pos and neg
accuracy =  0.8673469387755102
True positive rate /sensitivity, recall, hit rate =  0.908163265306
True negative rate specificity 0.0918367346939
precision =  0.839622641509
f1_score =  0.872549019608
False Discovery Rate 0.160377358491
false_positive =  0.173469387755

creditcard train dataset with full depth pos and neg is 20000
accuracy =  0.9966449094125541
True positive rate /sensitivity, recall, hit rate =  0.911688311688
True negative rate specificity 0.0883116883117
precision =  0.943548387097
f1_score =  0.927344782034
False Discovery Rate 0.0564516129032
false_positive =  0.00131184407796

creditcard test dataset with full depth pos and neg is 20000
accuracy =  0.9924353343094192
True positive rate /sensitivity, recall, hit rate =  0.841121495327
True negative rate specificity 0.158878504673
precision =  0.865384615385
f1_score =  0.85308056872
False Discovery Rate 0.134615384615
false_positive =  0.00350789275871

adult train dataset with full depth 
accuracy =  0.8944386090994527
True positive rate /sensitivity, recall, hit rate =  0.947583314942
True negative rate specificity 0.0524166850585
precision =  0.914970697922
f1_score =  0.930991489131
False Discovery Rate 0.0850293020778
false_positive =  0.266236654804

adult test dataset with full depth 
accuracy =  0.8329279080053074
True positive rate /sensitivity, recall, hit rate =  0.903925014646
True negative rate specificity 0.0960749853544
precision =  0.878326455102
f1_score =  0.890941898232
False Discovery Rate 0.121673544898
false_positive =  0.38583032491
'''
