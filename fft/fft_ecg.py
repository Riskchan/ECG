import wfdb
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import fftpack

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
N = 2**14
#dirs = getDirectories("data")
#for dir in dirs:
#    files = fnmatch.filter(os.listdir(dir), "*.dat")
#    for file in files:
#        filename = file.split(".")[0]
#        sig, fields=wfdb.rdsamp("data/" + dir + "/" + filename, channels=[0], sampfrom=0, sampto=N-1)

sig, fields=wfdb.rdsamp("data/patient001/s0010_re", channels=[5], sampfrom=0, sampto=N)
#sig, fields=wfdb.rdsamp("data/patient150/s0287lre", channels=[2], sampfrom=0, sampto=N)

#print(sig)

# FFT
dt = 0.001
t = np.linspace(1, N, N)*dt-dt

#f = 1
#y = np.sin(2*np.pi*f*t)

y = []
for i in range(N):
    y.append(sig[i][0])

yf = fftpack.fft(y)
freq = fftpack.fftfreq(N, dt)

# inverse fft
#sig2 = np.real(fftpack.ifft(yf))

plt.figure(1)
plt.plot(t, y)
plt.xlabel("time")
plt.ylabel("ECG")

plt.figure(3)
plt.subplot(211)
plt.plot(freq[1:N/16], np.abs(yf[1:N/16]))
plt.axis('tight')
plt.ylabel("Amplitude")
plt.subplot(212)
plt.plot(freq[1:N/16], np.degrees(np.angle(yf[1:N/16])))
plt.axis('tight')
plt.xlabel("Frequency")
plt.ylabel("phase[deg]")

plt.show()
#wfdb.plotwfdb(sig, fields)
