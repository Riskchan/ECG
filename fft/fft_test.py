# coding: utf-8
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 時系列のサンプルデータ作成
N = 512                         # データ数
dt = 0.01                       # サンプリング間隔
f = 5                          # 周波数
t = np.linspace(1, N, N)*dt-dt
y = np.sin(2*np.pi*f*t)

# 離散フーリエ変換
yf = fftpack.fft(y)
freq = fftpack.fftfreq(N, dt)

# プロット
# 時系列データ
plt.figure(1)
plt.plot(t, y)
plt.xlim(0, 1)
plt.xlabel("time")
plt.ylabel("y")

# 離散フーリエ変換の結果
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(np.linspace(1, N, N), np.real(yf), np.imag(yf))
ax.set_xlabel("data number")
ax.set_ylabel("real part")
ax.set_zlabel("imaginary part")

# 大きさ、位相
plt.figure(3)
plt.subplot(211)
#plt.plot(np.linspace(1, N, N), np.abs(yf))
plt.plot(freq, np.abs(yf))
plt.axis('tight')
plt.ylabel("amplitude")
plt.subplot(212)
plt.plot(freq, np.degrees(np.angle(yf)))
plt.axis('tight')
plt.xlabel("data number")
plt.ylabel("phase[deg]")

plt.show()
