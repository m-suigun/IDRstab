import numpy as np
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class ALE(object):
    def __init__(self, step=0.1, order=20, delay=10):
        self.step = step
        self.delaybuf = np.zeros(delay)
        self.firbuf = np.zeros(order)
        self.fircoef = np.zeros(order)


    def update_buf(self, buf, new):
        buf = shift(buf, 1)
        buf[0] = new
        return buf

    def delayed_signal(self, new):
        self.delaybuf = self.update_buf(self.delaybuf, new)
        return self.delaybuf[-1]

    def fir(self, new):
        self.firbuf = self.update_buf(self.firbuf, new)
        return np.dot(self.fircoef, self.firbuf)

    def update_coef(self, e):
        self.fircoef += (self.step / (0.000001 + np.dot(self.firbuf, self.firbuf))) * self.firbuf * e
        return 

    def partial_fit(self, new, ref):
        estimate = self.fir(self.delayed_signal(ref))
        error = new - estimate
        self.update_coef(error)
        return estimate, error

    def fit(self, inputs):
        y = np.zeros_like(inputs)
        e = np.zeros_like(inputs)
        for i in range(inputs.size):
            y[i], e[i] = self.partial_fit(inputs[i], inputs[i])
        return y, e



N = 5000
omega1 = 2.0 * np.pi * 0.05
omega2 = 2.0 * np.pi * 0.3

t = [i for i in range(N)]

np.random.seed(seed=32)
w = np.random.normal(loc=0.0, scale=1.0, size=N)
ar = np.zeros_like(w)
for i in range(N):
    ar[i] = 0.3 * ar[i-1] - 0.4 * ar[i-2] + w[i]
s = [np.sin(omega1 * i) + np.sin(omega2 * i) for i in range(N)]
x = np.array([s[i] + ar[i] for i in range(N)])


fir_order = 100
ale = ALE(step=0.1, order=fir_order, delay=20)
y, e = ale.fit(x)

plt.plot(t, x)
plt.plot(t, s)
plt.plot(t, y)
plt.show()

Nh = fir_order
padded = np.zeros(Nh)
for i in range(ale.fircoef.size):
    padded[i] = ale.fircoef[i]

H = np.fft.fft(padded)
fh = [i/Nh for i in range(Nh)]


X = np.fft.fft(x)
Y = np.fft.fft(y)
f = [i/N for i in range(N)]

plt.plot(f, np.abs(X)/N)
plt.plot(f, np.abs(Y)/N)
plt.plot(fh, np.abs(H))
plt.xlim(0, f[-1]/2)
plt.show()



E = np.fft.fft(e[1000:])
Ne = E.size
fe = [i/Ne for i in range(Ne)]
plt.plot(f, np.abs(X)/N)
plt.plot(fe, np.abs(E)/Ne)
plt.ylim(0, 0.1)
plt.xlim(0, f[-1]/2)
plt.show()

#plt.plot(ale.fircoef)
#plt.show()

estimate = np.zeros(N)
buf = np.zeros_like(ale.fircoef)
for i in range(N):
    buf = shift(buf, 1)
    buf[0] = w[i]
    estimate[i] = np.dot(ale.fircoef, buf)

#plt.plot(t, estimate)
#plt.plot(t, s)
#plt.show()