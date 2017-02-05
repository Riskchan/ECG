import wfdb
import os
import fnmatch
import numpy as np
import random
import tensorflow as tf

def getDirectories(path):
    dirs = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            dirs.append(item)
    return dirs

###############################################################################
# PTBDB
# Max signal length = 120012
# Min signal length = 32000
# N = 2^14 = 16384 would be appropriate

# Healthy control:          80
# Myocardial infarction:    368
# Dysrhythmia:              16
# Bundle branch block:      17
# Cardiomyopathy:           17
# Hypertrophy:              7
# Valvular heart disease:   6
# Myocarditi:               4
# Stable angina:            2
# Unstable angina:          1
# Palpitation:              1
# Heart failure (NYHA 2):   1
# Heart failure (NYHA 3):   1
# Heart failure (NYHA 4):   1
# n/a:                      27
###############################################################################
N = 2**14
diseases_cnt = {'Bundle branch block': 17, 'Valvular heart disease': 6, 'Stable angina': 2, 'Cardiomyopathy': 17, 'Dysrhythmia': 16, 'Heart failure (NYHA 2)': 1, 'Hypertrophy': 7, 'n/a': 27, 'Palpitation': 1, 'Heart failure (NYHA 4)': 1, 'Unstable angina': 1, 'Heart failure (NYHA 3)': 1, 'Healthy control': 80, 'Myocardial infarction': 368, 'Myocarditi': 4}
diseases_order = {'Bundle branch block': 0, 'Valvular heart disease': 1, 'Stable angina': 2, 'Cardiomyopathy': 3, 'Dysrhythmia': 4, 'Heart failure (NYHA 2)': 5, 'Hypertrophy': 6, 'n/a': 7, 'Palpitation': 8, 'Heart failure (NYHA 4)': 9, 'Unstable angina': 10, 'Heart failure (NYHA 3)': 11, 'Healthy control': 12, 'Myocardial infarction': 13, 'Myocarditi': 14}

num_keys = len(diseases_cnt.keys()) # number of diseases related to cardiovascular diseases

def getDiseaseVector(key):
    t = [0]*num_keys
    i = diseases_order[key]
    t[i] = 1
    return t

# Preparation for tensorflow training
sess = tf.InteractiveSession()

# Input
x = tf.placeholder(tf.float32, [None, N])

# Output layer
w = tf.Variable(tf.zeros([N, num_keys]))
w0 = tf.Variable(tf.zeros([num_keys]))
f = tf.matmul(x, w) + w0
p = tf.nn.softmax(f)

# Optimizer
t = tf.placeholder(tf.float32, [None, num_keys])
#loss = -tf.reduce_sum(t*tf.log(p))
loss = tf.reduce_sum(tf.square(p-t))
#train_step = tf.train.AdamOptimizer().minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

# run through training data
dirs = getDirectories("data")
diseases = {}

train_data = []
train_label = []
for dir in dirs:
    files = fnmatch.filter(os.listdir("data/" + dir), "*.dat")
    for file in files:
        # Read ptbdb
        filename = file.split(".")[0]
        sig, fields=wfdb.rdsamp("data/" + dir + "/" + filename, channels=[0], sampfrom=0, sampto=N)

        # Reason for admission
        reason = fields["comments"][4][23:]
        if reason in diseases:
            diseases[reason] += 1
        else:
            diseases[reason] = 1

        # Use half of the diseases as training resources
        # Use rest half of the diseases as testing resources
        #if diseases[reason] > diseases_cnt[reason]/2:
        #    continue

        # Reshape input ECG data
        train_x = np.zeros(N)
        for i in range(N):
            train_x[i] = sig[i][0]

        train_data.append(train_x)

        # get [0,0,....,1,0,....0] vector
        train_t = getDiseaseVector(reason)
        train_label.append(train_t)

# convert into numpy array
train_data = np.asarray(train_data)
train_label = np.asarray(train_label)

# Start learning
STEPS = 10000
BATCH_SIZE = 5
for i in range(STEPS):
        random_seq = range(len(train_data))
        random.shuffle(random_seq)

        for j in range(len(train_data)/BATCH_SIZE):
            batch = BATCH_SIZE * j
            train_data_batch = []
            train_label_batch = []

            for k in range(BATCH_SIZE):
                train_data_batch.append(train_data[random_seq[batch + k]])
                train_label_batch.append(train_label[random_seq[batch + k]])

        sess.run(train_step, feed_dict={x:train_data_batch, t:train_label_batch})

        # calculate loss and accuracy in each step
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:train_data, t:train_label})
        print("Loss:%g, Acc:%g"%(loss_val, acc_val))


#sig, fields=wfdb.rdsamp("data/patient001/s0010_re", channels=[0], sampfrom=0, sampto=N)
#print(fields["comments"][4])
#wfdb.plotwfdb(sig, fields)
