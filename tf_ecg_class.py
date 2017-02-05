import numpy as np
import random
import tensorflow as tf
from tf_ecgnetwork import tfECGNetwork
from tf_generate_tfdata import tfGenerateTFData

Num_input = 2**14

# Data
tfdata = tfGenerateTFData()
num_keys = tfdata.getNumberOfDiseases()
train_data, train_label = tfdata.getDatasets(Num_input)

# TF training
nn = tfECGNetwork(Num_input, num_keys)

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

        nn.sess.run(nn.train_step, feed_dict={nn.x:train_data_batch, nn.t:train_label_batch})

        # calculate loss and accuracy in each step
        if i%100 == 0:
            summary, loss_val, acc_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy],
                                                    feed_dict={nn.x:train_data, nn.t:train_label})
            print("Loss:%g, Acc:%g"%(loss_val, acc_val))
            nn.writer.add_summary(summary, i)


#sig, fields=wfdb.rdsamp("data/patient001/s0010_re", channels=[0], sampfrom=0, sampto=N)
#print(fields["comments"][4])
#wfdb.plotwfdb(sig, fields)
