import wfdb
import os
import fnmatch

def getDirectories(path):
    dirs = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            dirs.append(item)
    return dirs

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
# Unstable angina':         1
# Palpitation:              1
# Heart failure (NYHA 2):   1
# Heart failure (NYHA 3):   1
# Heart failure (NYHA 4):   1
# n/a:                      27

N = 2**14
diseases = {}
dirs = getDirectories("data")
for dir in dirs:
    files = fnmatch.filter(os.listdir("data/" + dir), "*.dat")
    for file in files:
        # Read ptbdb
        filename = file.split(".")[0]
        sig, fields=wfdb.rdsamp("data/" + dir + "/" + filename, channels=[1], sampfrom=0, sampto=N)

        # Reason for admission
        reason = fields["comments"][4][23:]
        if reason in diseases:
            diseases[reason] += 1
        else:
            diseases[reason] = 1

print(diseases)

#sig, fields=wfdb.rdsamp("data/patient001/s0010_re", channels=[0], sampfrom=0, sampto=N)
#print(fields["comments"][4])
#wfdb.plotwfdb(sig, fields)
