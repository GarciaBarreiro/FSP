from matplotlib import pyplot as plt
import numpy as np


def main():
    x = np.arange(-10,10,0.1)
    y = np.exp(-x**2)

    plt.plot(x,y)
    plt.grid()
    plt.title('Integral de Gauss')
    plt.savefig("Integral.png")

if __name__ == "__main__":
    main()
