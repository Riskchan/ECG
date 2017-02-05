import wfdb
import os
import fnmatch
import numpy as np

###############################################################################
# PTBDB
# Max signal length = 120012
# Min signal length = 32000
# N = 2^14 = 16384 would be appropriate for FFT

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

class tfGenerateTFData:
    def __init__(self):
        self.diseases_cnt = {'Bundle branch block': 17, 'Valvular heart disease': 6, 'Stable angina': 2, 'Cardiomyopathy': 17, 'Dysrhythmia': 16, 'Heart failure (NYHA 2)': 1, 'Hypertrophy': 7, 'n/a': 27, 'Palpitation': 1, 'Heart failure (NYHA 4)': 1, 'Unstable angina': 1, 'Heart failure (NYHA 3)': 1, 'Healthy control': 80, 'Myocardial infarction': 368, 'Myocarditi': 4}
        self.diseases_order = {'Bundle branch block': 0, 'Valvular heart disease': 1, 'Stable angina': 2, 'Cardiomyopathy': 3, 'Dysrhythmia': 4, 'Heart failure (NYHA 2)': 5, 'Hypertrophy': 6, 'n/a': 7, 'Palpitation': 8, 'Heart failure (NYHA 4)': 9, 'Unstable angina': 10, 'Heart failure (NYHA 3)': 11, 'Healthy control': 12, 'Myocardial infarction': 13, 'Myocarditi': 14}
        self.num_keys = len(self.diseases_cnt.keys()) # number of diseases related to cardiovascular diseases

    def getDirectories(self, path):
        dirs = []
        for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                dirs.append(item)
        return dirs

    def getNumberOfDiseases(self):
        return self.num_keys

    def getDiseaseVector(self, key):
        t = [0]*self.num_keys
        i = self.diseases_order[key]
        t[i] = 1
        return t

    def getDatasets(self, N):
        # run through training data
        dirs = self.getDirectories("data")
        diseases = {}

        train_data = []
        train_label = []
        for dir in dirs:
            files = fnmatch.filter(os.listdir("data/" + dir), "*.dat")
            for file in files:
                # Read ptbdb
                filename = file.split(".")[0]
                sig, fields = wfdb.rdsamp("data/" + dir + "/" + filename, channels=[0], sampfrom=0, sampto=N)

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
                train_t = self.getDiseaseVector(reason)
                train_label.append(train_t)

        # convert into numpy array
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label)

        return train_data, train_label
