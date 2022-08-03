
from attr import dataclass
import numpy as np
import os
from nltk import ngrams
from pandas.core.frame import DataFrame
import os
import time
import random
import pickle
import math
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from keras.layers import Dropout
from tensorflow import keras
from collections import Counter
from collections import defaultdict  
from datetime import date  
import random

#Setting pointing to where one wants to load and save data. 
os.chdir("/home/ubuntu/jesse/MaskedEventAnomalyDetection/data")

#Global variables
_ngrams_ = 5
_start_ ="SoS" #Start of Sequence used in padding the sequence
_end_ = "EoS" #End of Sequence used in padding the sequence
predictword = "next"   #options "next" "masked"
#More clo
n_gram_counter = Counter()
n_gram_counter_1 = Counter() 
n1_gram_dict = defaultdict() # to keep mappings of possible following events e1 e2 -> e1 e2 e3, e1 e2 e4, 
n1_gram_winner = dict() #What is the event n following n-1 gram, i.e. the prediction ?


def create_ngram_model(train_data):
    global n_gram_counter, n_gram_counter_1
    ngrams = list()
    ngrams_1 = list()
    for seq in train_data:
        seqs, seqs_1 = slice_to_ngrams(seq)
        ngrams.extend(seqs)
        ngrams_1.extend(seqs_1)
    n_gram_counter += Counter (ngrams)
    n_gram_counter_1 += Counter (ngrams_1)

    for idx, s in enumerate(ngrams):
        #dictionary for faster access from n-1 grams to n-grams, e.g. from  [e1 e2 e3] -> [e1 e2 e3 e4]; [e1 e2 e3] -> [e1 e2 e3 e5] etc...
        n1_gram_dict.setdefault(ngrams_1[idx],[]).append(s)
        #precompute the most likely sequence following n-1gram. Needed to keep prediction times fast
        if (ngrams_1[idx] in n1_gram_winner): #is there existing winner 
            n_gram = n1_gram_winner[ngrams_1[idx]]
            if (n_gram_counter[n_gram] < n_gram_counter[s]): #there is but we are bigger replace
                n1_gram_winner[ngrams_1[idx]] = s
        else: 
            n1_gram_winner[ngrams_1[idx]] = s #no n-1-gram key or winner add a new one...

#Produce required n-grams. E.g. With sequence [e1 ... e5] and _ngrams_=3 we produce [e1 e2 e3], [e2 e3 e4], and [e3 e4 5] 
def slice_to_ngrams (seq):
    #Add SoS and EoS
    #with n-gram 3 it is SoS SoS E1 E2 E3 EoS
    #No need to pad more than one EoS as the final event to be predicted is EoS
    if (padding):
        if predictword=="next":
            seq = [_start_]*(_ngrams_-1) +seq+[_end_]
        #Calculate appropriate amount of padding based on the predict position
        elif predictword=="masked":
            seq = [_start_]*(_ngrams_-1-predfromlast) +seq+[_end_]*(1+predfromlast)

    ngrams = list()
    ngrams_1 = list()
    for i in range(_ngrams_, len(seq)+1):#len +1 because [0:i] leaves out the last element 
        ngram_s = seq[i-_ngrams_:i]
        # convert into a line
        line = ' '.join(ngram_s)
        # store
        ngrams.append(line)
        ngram_s_1= seq[i-_ngrams_:i-1] #if i=13, takes from indexes 8 to 11
        #masked word test
        if(predictword == "masked"):
            temp = seq[i-_ngrams_:i-(predfromlast+1)]
            temp.extend(seq[i-_ngrams_+(_ngrams_-predfromlast):i])
            ngram_s_1 = temp

        line2 = ' '.join(ngram_s_1)
        # store
        ngrams_1.append(line2)
    return ngrams, ngrams_1


def load_pro_data():
    pro_x = np.load("profilence_x_data.npy", allow_pickle=True)
    pro_y = np.load("profilence_y_data.npy", allow_pickle=True)

    pro_y = pro_y == 1
    abnormal_test = pro_x[pro_y]

    pro_x_normal = pro_x[~pro_y]
    from nltk import ngrams

    lengths = list()
    for seq in pro_x_normal:
        lengths.append(len(seq))
    #zeros = np.array([True if i ==0 else False for i in lengths])
    #pro_x_normal = pro_x_normal[~zeros]
    #Remove the short logs less than 10000
    ten_k_lenght = np.array([True if i >= 10000 else False for i in lengths])
    pro_x_normal = pro_x_normal[ten_k_lenght]
    normal_data = pro_x_normal
    return normal_data, abnormal_test


def load_hdfs_data():
    hdfs_x = np.load("hdfs_x_data.npy", allow_pickle=True)
    hdfs_y = np.load("hdfs_y_data.npy", allow_pickle=True)

    hdfs_y = hdfs_y == 1

    hdfs_x_normal = hdfs_x[~hdfs_y]
    abnormal_test = hdfs_x[hdfs_y]
    normal_data = hdfs_x_normal
    return normal_data, abnormal_test


def load_hadoop_data():
    hadoop_x = np.load("hadoop_x_data.npy", allow_pickle=True)
    hadoop_y = np.load("hadoop_y_data.npy", allow_pickle=True)

    hadoop_y = hadoop_y == 1

    hadoop_x_normal = hadoop_x[~hadoop_y]
    abnormal_test = hadoop_x[hadoop_y]
    normal_data = hadoop_x_normal
    
    for x in range(0, len(normal_data)):
        normal_data[x] = normal_data[x].tolist()

    for x in range(0, len(abnormal_test)):
        abnormal_test[x] = abnormal_test[x].tolist()

    return normal_data, abnormal_test


def load_bgl_data():
    #bgl_x = np.load("bgl_x_data.npy", allow_pickle=True)
    #bgl_y = np.load("bgl_y_data.npy", allow_pickle=True)
    bgl_x = np.load("bgl_x_seq_"+str(_ngrams_)+".npy", allow_pickle=True)
    bgl_y = np.load("bgl_y_seq_"+str(_ngrams_)+".npy", allow_pickle=True)

    bgl_y = bgl_y == 1

    bgl_x_normal = bgl_x[~bgl_y]
    abnormal_test = bgl_x[bgl_y]
    normal_data = bgl_x_normal

    return normal_data, abnormal_test


#Reset global n-gram variables. Used when creating multiple n-gram models
def reset_globals():
    global n_gram_counter, n_gram_counter_1, n1_gram_dict, n1_gram_winner
    n_gram_counter = Counter()
    n_gram_counter_1 = Counter()
    from collections import defaultdict
    n1_gram_dict = defaultdict() # to keep mappings of possible following events e1 e2 -> e1 e2 e3, e1 e2 e4, 
    n1_gram_winner = dict()
    #sequences = list()
    #sequences_1 = list()



def create_LSTM_model(ngrams, vocab_size, share_of_data=1,  model_epoch=10):
    #If we want to use less than 100% of data select samples. I am not sure this is ever used
    if (share_of_data < 1):
        select = int(len(ngrams) * share_of_data)
        ngrams = random.sample(ngrams, select)
    random.shuffle(ngrams)
    # How many dimensions will be used to represent each event. 
    # With words one would use higher values here, e.g. 200-400
    # Higher values did not improve accuracy but did reduce perfomance. Even 50 might be too much
    dimensions_to_represent_event = 50

    #For early stopping, not used at the moment
    callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=2, min_delta=0.01)
    opt = keras.optimizers.Adam() #learning_rate=0.01

    model = Sequential()
    model.add(Embedding(vocab_size, dimensions_to_represent_event, input_length=_ngrams_-1))
    # We will use a two LSTM hidden layers with 100 memory cells each. 
    # More memory cells and a deeper network may achieve better results.
    model.add(LSTM(100, return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 

    #Loop needed as Office PC would crash in the to_categorixal with Profilence data set as it out of memory. 
    #TODO: Do we need a loop when using CSC HW?
    loop_variable = 500000000
    for x in range(0, len(ngrams), loop_variable):
        print(f'loop with x= {x}. / {len(ngrams)}')
        ngrams0 = np.array(ngrams[x:x+loop_variable])

        if(predictword == "next"):
            X, y = ngrams0[:,:-1], ngrams0[:,-1]
        #masked word test
        if(predictword == "masked"):
            X = ngrams0[:,:(-1-predfromlast)]
            if(predfromlast > 0):
                X = np.append(X, ngrams0[:,-predfromlast:],axis=1)
            y = ngrams0[:,(-1-predfromlast)]

        y = to_categorical(y, num_classes=vocab_size)
        #Modify batch_size and epoch to influence the training time and resulting accuracy. 
        history = model.fit(X, y, validation_split=0.05, batch_size=1024, epochs=model_epoch, shuffle=True).history
    
    return model

def create_CNN_model(ngrams, vocab_size, share_of_data=1, model_epoch=10):
    #If we want to use less than 100% of data select samples. I am not sure this is ever used
    if (share_of_data < 1):
        select = int(len(ngrams) * share_of_data)
        ngrams = random.sample(ngrams, select)
    random.shuffle(ngrams)
    # How many dimensions will be used to represent each event. 
    # With words one would use higher values here, e.g. 200-400
    # Higher values did not improve accuracy but did reduce perfomance. Even 50 might be too much
    dimensions_to_represent_event = 50

    #For early stopping, not used at the moment
    callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, min_delta=0.001)
    opt = keras.optimizers.Adam() #learning_rate=0.01
    
    model = Sequential()
    model.add(Embedding(vocab_size, dimensions_to_represent_event, input_length=_ngrams_-1))
    #model.add(Dropout(0.2, input_shape=(100,)))

    # We will use a two LSTM hidden layers with 100 memory cells each. 
    # More memory cells and a deeper network may achieve better results.
    #model.add(LSTM(100, return_sequences=True))
    #model.add(LSTM(100))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(vocab_size, activation='softmax'))
    #model.add(Conv1D(filters=20 , kernel_size = _ngrams_-1, activation='relu', input_shape=(_ngrams_ -1 , 1)))
    #model.add(GlobalMaxPooling1D())
    model.add(Conv1D(filters=20 , kernel_size = _ngrams_-1, activation='relu'))
    model.add(GlobalMaxPooling1D())
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
   # print(model.summary())
    
    #Loop needed as Office PC would crash in the to_categorixal with Profilence data set as it out of memory. 
    #TODO: Do we need a loop when using CSC HW?
    loop_variable = 50000000
    for x in range(0, len(ngrams), loop_variable):
        print(f'loop with x= {x}. / {len(ngrams)}')
        ngrams0 = np.array(ngrams[x:x+loop_variable])
        if(predictword == "next"):
            X, y = ngrams0[:,:-1], ngrams0[:,-1]
        #masked word test
        if(predictword == "masked"):
            X = ngrams0[:,:(-1-predfromlast)]
            if(predfromlast > 0):
                X = np.append(X, ngrams0[:,-predfromlast:],axis=1)
            y = ngrams0[:,(-1-predfromlast)]

        y = to_categorical(y, num_classes=vocab_size)
        #Modify batch_size and epoch to influence the training time and resulting accuracy. 
        history = model.fit(X, y, validation_split=0.05, batch_size=1024, epochs=model_epoch, shuffle=True).history #callbacks=[callback],
    
    return model


# We need to change events e1 e2 e3 to numbers for the DL model so they are mapped here, e.g. e1 -> 137, e2 -> 342 
def sequences_to_dl_ngrams (train_data):
    ngrams = list() #ngrams= []
    for seq in train_data:
        t_ngrams, t_ngrams_1 = slice_to_ngrams(seq)
        ngrams.extend(t_ngrams)
    tokenizer = Tokenizer(oov_token=1)    
    tokenizer.fit_on_texts(ngrams)
    ngrams_num = tokenizer.texts_to_sequences(ngrams)
    vocab_size = len(tokenizer.word_index) + 1
    return ngrams, ngrams_num, vocab_size, tokenizer

#Gives N-gram predictions
def give_preds (seq):
    seq_shingle, seq_shingle_1 = slice_to_ngrams(seq)
    #   print(seq_shingle)
    correct_preds = list()
    pred_values = list()
    for s in seq_shingle:
        to_be_matched_s =  s.rpartition(' ')[0]
        if (predictword == "masked" ):
            splitted = s.split(' ')
            asplit = splitted[:(_ngrams_-predfromlast-1)]
            asplit.extend(splitted[(_ngrams_-predfromlast):]) 
            to_be_matched_s = " ".join(asplit)
        #print("to be matched " + to_be_matched_s)
        if (to_be_matched_s in n1_gram_dict):
            winner = n1_gram_winner[to_be_matched_s]
            if (winner == s):
                correct_preds.append(1)
                #value of predicted event, i.e. "A0010"
                if(predictword == "masked" ):
                    pred_values.append(splitted[_ngrams_-1-predfromlast])
                if(predictword == "next" ):
                    pred_values.append(s.rpartition(' ')[2])
                #print("correct")
            else: 
                correct_preds.append(0)
                #print("incorrec predic")
        else:
            correct_preds.append(0)
            #print("no key")
    return correct_preds, pred_values

#LSTM prediction per sequence. Typically called from loop that with HDFS is not efficient
def give_preds_lstm (seq):
    seq_shingle, seq_shingle_1 = slice_to_ngrams(seq)
    seq_shingle_num = lstm_tokenizer.texts_to_sequences(seq_shingle)
    seq_shingle_num_np = np.array(seq_shingle_num)
    seq_shingle_num_1 = seq_shingle_num_np[:,:-1]
    seq_shingle_truth = seq_shingle_num_np[:,-1]

    #predicted_sec = model.predict(seq_shingle_num_1)
    predicted_sec = model.predict(seq_shingle_num_1,verbose=1, batch_size=4096)
    predicted_events = np.argmax(predicted_sec, axis=1)

    correct_preds = seq_shingle_truth == predicted_events
    return correct_preds

#LSTM predictions with multiple sequences packed in numpy array 
def give_preds_lstm_2 (sequences, b_size=4096):
    seq_shingle = list()
    #check if this is an array of sequences
    start_s = time.time()
    if (isinstance(sequences, np.ndarray)):
        for s in sequences:
            temp_seq_shingle, temp_seq_shingle_1 = slice_to_ngrams(s)
            seq_shingle.extend(temp_seq_shingle)
    else: #if not numpy array then as 
        seq_shingle, seq_shingle_1 = slice_to_ngrams(sequences)
    end_s = time.time()
    print("Shingle creation took", end_s - start_s)
    start_s = time.time()
    seq_shingle_num = lstm_tokenizer.texts_to_sequences(seq_shingle) #do this before slice to n-grams
    end_s = time.time()
    print("lstm_tokenizer took", end_s - start_s)
    seq_shingle_num_np = np.array(seq_shingle_num)

    if(predictword == "next"):
        seq_shingle_num_1 = seq_shingle_num_np[:,:-1]
        seq_shingle_truth = seq_shingle_num_np[:,-1]
    if (predictword == "masked"):
        seq_shingle_num_1 = seq_shingle_num_np[:,:(-1-predfromlast)]
        if (predfromlast > 0):
            seq_shingle_num_1 = np.append(seq_shingle_num_1, seq_shingle_num_np[:,-predfromlast:], axis=1)
        seq_shingle_truth = seq_shingle_num_np[:,(-1-predfromlast)]

    #predicted_sec = model.predict(seq_shingle_num_1)
    start_s = time.time()
    predicted_sec = model.predict(seq_shingle_num_1,verbose=0, batch_size=b_size)
    end_s = time.time()
    print("prediction took", end_s - start_s)
    #predicted_sec = model.predict(seq_shingle_num_1, verbose=1, use_multiprocessing = True, max_queue_size=100,workers=4)
    predicted_events = np.argmax(predicted_sec, axis=1)

    correct_preds = seq_shingle_truth == predicted_events
    return correct_preds


#Prints sheet format from global variables. Example (tabs adjusted for readability):
#Data	Method	predict word  Window (n)	HW	Measure	    Split	Source	        Measurement
#HDFS	CNN	    masked        5		        Accuracy        50-50	M-23-02-2022	0.847
#gets other than method, measure and measurement from global variables!

def sheet_form_print(method, measure, measurement): 
    source = "J-"+date_str #initial letter from name + date comes from variable
    hw = "GPU-PC" #GPU-PC is CSC
    #make data source text uniform
    datastr = data.capitalize()
    if(data == "hdfs" or data == "bgl"):
        datastr = data.upper()
    #if masked, add position of predictionS
    predstr = predictword
    if(predstr=="masked"):
        predstr = "masked_"+str(predfromlast)
    sheetstr = datastr+"\t"+method+"\t"+predstr+"\t"+str(_ngrams_)+"\t"+hw+"\t"+measure+"\t"+str(split)+"-"+str(100-int(split))+"\t"+source+"\t"+str(round(measurement,3))
    print(sheetstr)
    #save results to file
    with open('results_'+source+'.txt', 'a') as f:
                f.write(sheetstr+"\n")


def ngram_prediction(normal_test):
#ngram prediction-------------------------------------------
    #ngram test with loop
    ngram_preds = list()
    ngram_preds2 = list()
    ngram_predvalues = list()
    start_s = time.time()
    for normal_s in normal_test:
        preds, values = give_preds(normal_s)
        ngram_preds.append(preds)
        ngram_preds2.extend(preds)
        ngram_predvalues.extend(values)
        #print(".")
    end_s = time.time()
    #print("prediction time ngram with ngrams:", _ngrams_, "done in", end_s - start_s)
    sheet_form_print("N-Gram", "I-Time", end_s - start_s)
    #ngram investigate
    ngram_preds_means = list()
    for preds in ngram_preds:
        ngram_mean = np.mean(preds)
        ngram_preds_means.append(ngram_mean)
        #print (np.mean(lstm_mean))

    valuedf = DataFrame(ngram_predvalues)
    sheet_form_print("N-Gram", "Accuracy (Mom)", np.mean(ngram_preds_means))
    sheet_form_print("N-Gram", "Accuracy (Moa)", np.mean(ngram_preds2))
    #print("Most frequent event: "+ valuedf.value_counts(normalize=True).index[0][0] +", "+ str(round(valuedf.value_counts(normalize=True)[0], 3)*100) + "%")
    #print("Mean of means", np.mean(ngram_preds_means))
    #print("Mean of all", np.mean(ngram_preds2))


#No need to run each time
#Create multiple .txt files for the splits between train and test data based on dataset names and portions
def create_splits(datasets = ["pro","hdfs", "bgl", "hadoop"] ):
    portions = [10,25,50,75,90]

    for dataset in datasets:
        funcstr = "load_"+dataset+"_data()"
        normal_data = eval(funcstr)[0]
        
        for portion in portions:
            train_i = np.random.choice(normal_data.shape[0], int(normal_data.shape[0]*(portion/100)), replace=False)
            normal_train = np.isin(range(normal_data.shape[0]), train_i)
            namestr = "split_"+str(portion)+"_"+dataset+"_train.txt"
            np.savetxt(namestr, normal_train, fmt='%d')

def sequences_total_and_unique():
    datasets = ["pro","hdfs", "bgl", "hadoop"] 
    split = 50
    for data in datasets:
        reset_globals() 
        normal_train = np.loadtxt('split_' + str(split) + '_' + data + '_train.txt') #load split
        funcstr = "load_" + data + "_data()" 
        normal_data,abnormal_test = eval(funcstr)
        normal_train = np.array(normal_train, dtype=bool)
        normal_test = normal_data[~normal_train]
        print("Dataset: " + data + " total sequences::"+ str(len(normal_data))+", total training sequences: " + str(len(normal_data[normal_train])) + ", unique training sequences: " + str(len(np.unique(normal_data[normal_train]))))
    


# END of Functions-------------------------------------------------------------------------------------------------------------------

#Variables for both single and multiruns

padding = True

predictword = "next"    #determine which spot to predict. 0 = last. E.g. 2 is middle if _ngrams_ = 5
predictword = "masked" 

predfromlast = 0

_ngrams_ = 5

create_model = "no"
create_model = "yes"
for_training = 1

uniques = 0

model_type="lstm"
model_type="cnn"
date_str = "2022-06-17" 



#--------------- Single runs ---------------------

# Data loading
#Select variable on which data set to load
data="hdfs"
data="bgl"
data="pro"
data="hadoop"
split = "50" #new function generates number string, old have "normal"
print("Setting data to " + data + " with " + split + " split")
normal_train = np.loadtxt('split_'+split+'_'+data+'_train.txt') #load split

if (data=="hdfs"):
    normal_data, abnormal_test = load_hdfs_data() #load data
elif(data=="pro"):
    normal_data, abnormal_test = load_pro_data() #load data"
elif (data=="hadoop"):
    normal_data, abnormal_test = load_hadoop_data() #load data
elif (data=="bgl"):
    normal_data, abnormal_test = load_bgl_data() #load data

normal_train = np.array(normal_train, dtype=bool)
normal_test = normal_data[~normal_train]


#-N-Gram
start_s = time.time()
create_ngram_model(normal_data[normal_train])
end_s = time.time()
print("ngram with ngrams:", _ngrams_, "done in", end_s - start_s)
ngram_prediction(normal_test)

#LSTM Model creation
if (predictword=="next"):
    predfromlast = 0
if (create_model=="yes"):
    start_s = time.time()
    if (uniques):
        lstm_ngrams, lstm_ngrams_num, lstm_vocab_size, lstm_tokenizer = sequences_to_dl_ngrams(np.unique(normal_data[normal_train]))
    else:
        lstm_ngrams, lstm_ngrams_num, lstm_vocab_size, lstm_tokenizer = sequences_to_dl_ngrams(normal_data[normal_train])
    if (model_type=="lstm"):
        model = create_LSTM_model(lstm_ngrams_num, lstm_vocab_size, share_of_data=for_training)
    elif(model_type=="cnn"):
        model = create_CNN_model(lstm_ngrams_num, lstm_vocab_size, share_of_data=for_training)
    end_s = time.time()

    sheet_form_print(model_type.upper(), "T-Time", end_s - start_s)
    #print(model_type," with ngrams:", _ngrams_, "done in", end_s - start_s)
    model.save("model_"+model_type+"_"+str(_ngrams_)+"_"+str(predfromlast)+"_"+str(for_training)+"_"+data+"_"+str(date.today()))
    # saving tokenizer
    with open("tokenizer_"+model_type+"_"+str(_ngrams_)+"_"+str(predfromlast)+"_"+data+"_"+str(date.today())+'.pickle', 'wb') as handle:
        pickle.dump(lstm_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
elif(create_model=="no"):
    model = keras.models.load_model("model_"+model_type+"_"+str(_ngrams_)+"_"+str(for_training)+"_"+str(predfromlast)+"_"+data+"_"+date_str)
    with open("tokenizer_"+model_type+"_"+str(_ngrams_)+"_"+str(predfromlast)+"_"+data+"_"+date_str+'.pickle', 'rb') as handle:
        lstm_tokenizer = pickle.load(handle)
    

# LSTM Prediction-------------------------
#LSTM much faster with HDFS as one predict call instead of loop
start_s = time.time()
lstm_preds_all = list()
if (data!="pro"):
    lstm_preds_all = give_preds_lstm_2(normal_test)
else :#Cannot do all pro data in one pass runs out of memory at 15gigs. Split to five calls
    for i in range(int(len(normal_test)/10)):
        lstm_preds_t = give_preds_lstm_2(normal_test[i:i+10])
        lstm_preds_all.extend(lstm_preds_t)  
end_s = time.time()
sheet_form_print(model_type.upper(), "I-Time", end_s - start_s)
sheet_form_print(model_type.upper(), "Accuracy", np.mean(lstm_preds_all))
#np.mean(lstm_preds_all)




#------------------- Multi runs -------------------------------

#Run N-Gram model multiple times with given settings
datasets = ["hdfs"] #"bgl","pro","hadoop","hdfs"
seqlen = [5]
splits = [50] #Each split requires split text files
maskrange = range(0,1)
uniques = 1

for predfromlast in maskrange: 
    for _ngrams_ in seqlen:
        if predfromlast < _ngrams_:
            for data in datasets:
                print("Setting dataset to " + data + " - Uniques:" + str(uniques))
                for split in splits:
                    if (data=="bgl"):
                        padding = False
                        create_splits(["bgl"]) #different sequence lengths have own datasets -> new splits
                    else:
                        padding = True
                    reset_globals() 
                    normal_train = np.loadtxt('split_' + str(split) + '_' + data + '_train.txt') #load split
                    funcstr = "load_" + data + "_data()" 
                    normal_data,abnormal_test = eval(funcstr)
                    normal_train = np.array(normal_train, dtype=bool)
                    normal_test = normal_data[~normal_train]
                    start_s = time.time()
                    if (uniques):
                        create_ngram_model(np.unique(normal_data[normal_train]))
                    else:
                        create_ngram_model(normal_data[normal_train])
                    end_s = time.time()
                    sheet_form_print("N-Gram", "T-Time", end_s - start_s)
                    ngram_prediction(normal_test)


    #Multiruns for lstm and cnn
    datasets = ["hadoop"] #"bgl","pro","hadoop","hdfs"
    splits = [50] #[5,10,25,50,75,90,95] available
    seqlen = [5] #[2,5,10,15,20]
    maskrange = [0]  #[0,1,2,3,4]
    model_types = ["cnn"] #"lstm","cnn"
    epoch = 1449

    for model_type in model_types: 
        for uniques in range (1, 2):
            for predfromlast in maskrange: 
                for _ngrams_ in seqlen:
                    if predfromlast < _ngrams_:
                        for data in datasets:
                            print("Setting dataset to " + data + " - Uniques:" + str(uniques))                            
                            if (data=="bgl"):
                                padding = False
                                create_splits(["bgl"]) #different sequence lengths have own datasets -> new splits
                            else:
                                padding = True
                            for split in splits:
                                reset_globals()
                                funcstr = "load_" + data + "_data()" 
                                normal_data,abnormal_test = eval(funcstr)
                                normal_train = np.loadtxt('split_' + str(split) + '_' + data + '_train.txt') #load split
                                normal_train = np.array(normal_train, dtype=bool)
                                normal_test = normal_data[~normal_train]
                                
                                if (predictword=="next"):
                                    predfromlast = 0
                                if (create_model=="yes"):
                                    start_s = time.time()
                                    if (uniques):
                                        lstm_ngrams, lstm_ngrams_num, lstm_vocab_size, lstm_tokenizer = sequences_to_dl_ngrams(np.unique(normal_data[normal_train]))
                                    else:
                                        lstm_ngrams, lstm_ngrams_num, lstm_vocab_size, lstm_tokenizer = sequences_to_dl_ngrams(normal_data[normal_train])
                                    if (model_type=="lstm"):
                                        model = create_LSTM_model(lstm_ngrams_num, lstm_vocab_size, share_of_data=for_training, model_epoch=epoch)
                                    elif(model_type=="cnn"):
                                        model = create_CNN_model(lstm_ngrams_num, lstm_vocab_size, share_of_data=for_training, model_epoch=epoch)
                                    end_s = time.time()

                                    sheet_form_print(model_type.upper(), "T-Time", end_s - start_s)
                                    #print(model_type," with ngrams:", _ngrams_, "done in", end_s - start_s)
                                    model.save("model_"+model_type+"_"+str(_ngrams_)+"_"+str(predfromlast)+"_"+str(for_training)+"_"+data+"_"+str(date.today()))
                                    # saving tokenizer
                                    with open("tokenizer_"+model_type+"_"+str(_ngrams_)+"_"+str(predfromlast)+"_"+data+"_"+str(date.today())+'.pickle', 'wb') as handle:
                                        pickle.dump(lstm_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                elif(create_model=="no"):
                                    model = keras.models.load_model("model_"+model_type+"_"+str(_ngrams_)+"_"+str(for_training)+"_"+str(predfromlast)+"_"+data+"_"+date_str)
                                    with open("tokenizer_"+model_type+"_"+str(_ngrams_)+"_"+str(predfromlast)+"_"+data+"_"+date_str+'.pickle', 'rb') as handle:
                                        lstm_tokenizer = pickle.load(handle)
                                
                                # LSTM Prediction------------------------------------------------------------------
                                #LSTM much faster with HDFS as one predict call instead of loop
                                start_s = time.time()
                                lstm_preds_all = list()
                                if (data!="pro"):
                                    lstm_preds_all = give_preds_lstm_2(normal_test)
                                else :#Cannot do all pro data in one pass runs out of memory at 15gigs. Split to five calls
                                    for i in range(int(len(normal_test)/10)):
                                        lstm_preds_t = give_preds_lstm_2(normal_test[i:i+10])
                                        lstm_preds_all.extend(lstm_preds_t)  
                                end_s = time.time()
                                sheet_form_print(model_type.upper(), "I-Time", end_s - start_s)
                                sheet_form_print(model_type.upper(), "Accuracy", np.mean(lstm_preds_all))
                                #np.mean(lstm_preds_all)

