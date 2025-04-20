import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_cof_vA(n_A, n_B, n_E, lambda_B, lambda_EA):
    xi = lambda_B/lambda_EA
    tau = get_tau(n_A, n_B, n_E, xi)
    temp1 = n_A*(np.log2(lambda_EA) - np.log2(lambda_B))
    temp2 = n_B*np.log2(1+n_A*tau)+n_E*np.log2(1+xi*n_A*tau)
    temp3 = (n_E-n_A)*np.log2(np.max((n_E-n_A, 1))) - \
        n_E*np.log2(n_E)-n_A*np.log2(tau)
    return temp1+temp2+temp3


def get_cof_vB(n_A, n_B, n_E, lambda_A, lambda_EB):
    temp1 = n_B*(np.log2(lambda_EB)-np.log2(lambda_A))
    temp2 = (n_E-n_B)*np.log2(np.max((n_E-n_B, 1))) - \
        (n_A-n_B)*np.log2(np.max((n_A-n_B, 1)))
    temp3 = n_A*np.log2(n_A)-n_E*np.log2(n_E)
    return temp1+temp2+temp3


def get_tau(n_A, n_B, n_E, xi):
    delta_mu = n_A-n_B-n_E
    BB = (xi*(n_A-n_E)+(n_A-n_B))/(delta_mu*xi*n_A)
    CC = 1/(delta_mu*xi*n_A)
    tau1 = 0.5*(-BB+np.sqrt((BB**2)-(4*CC)))
    tau2 = 0.5*(-BB-np.sqrt((BB**2)-(4*CC)))
    tau = np.max((tau1, tau2))
    return tau


# %%
if __name__ == '__main__':

    n_A = 16
    n_B = 12
    lambda_B = 1.25
    lambda_EA = 0.5
    x = np.zeros(256-n_A)
    y = np.zeros(256-n_A)
    for i in range(256-n_A):
        x[i] = n_A+i
        y[i] = get_cof_vA(n_A, n_B, n_A+i, lambda_B, lambda_EA)
    my_label = '$n_A={},\;n_B={},\;\lambda_B={},\;\lambda_{{EA}}={}$'
    plt.semilogy(x, y, '-', color='magenta',
                 label= my_label.format(n_A, n_B, lambda_B, lambda_EA))

    n_A = 32
    n_B = 12
    lambda_B = 1.25
    lambda_EA = 0.5
    x = np.zeros(256-n_A)
    y = np.zeros(256-n_A)
    for i in range(256-n_A):
        x[i] = n_A+i
        y[i] = get_cof_vA(n_A, n_B, n_A+i, lambda_B, lambda_EA)
    plt.semilogy(x, y, '--', color='blue',
                  label=my_label.format(n_A, n_B, lambda_B, lambda_EA))

    plt.xlabel(r'$n_E$', fontsize=15)
    plt.ylabel(r'$\omega_{A}$', fontsize=15)
    plt.xlim([0, 256])
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(which='both', axis='both')
    plt.savefig('fig_vA_coeff.jpg', bbox_inches='tight',
                pad_inches=0.02, dpi=450)
    plt.show()

# %%
if __name__ == '__main__':

    n_A = 16
    n_B = 12
    lambda_A = 1.25
    lambda_EB = 0.5
    x = np.zeros(256-n_A)
    y = np.zeros(256-n_A)
    for i in range(256-n_A):
        x[i] = n_A+i
        y[i] = get_cof_vB(n_A, n_B, n_A+i, lambda_A, lambda_EB)
    my_label_1 = r'$n_A=$'+str(n_A)+r'$; n_B=$'+str(n_B)+r'; $\lambda_A=$' +\
        str(lambda_A)+r'; $\lambda_{EB}=$'+str(lambda_EB)
    plt.semilogy(x, -y, '-', color='magenta', label=my_label_1)

    n_A = 32
    n_B = 12
    lambda_A = 3
    lambda_EB = 1
    x = np.zeros(256-n_A)
    y = np.zeros(256-n_A)
    for i in range(256-n_A):
        x[i] = n_A+i
        y[i] = get_cof_vB(n_A, n_B, n_A+i, lambda_A, lambda_EB)
    my_label_2 = r'$n_A=$'+str(n_A)+r'$; n_B=$'+str(n_B)+r'; $\lambda_A=$' +\
        str(lambda_B)+r'; $\lambda_{EB}=$'+str(lambda_EB)
    plt.semilogy(x, -y, '--', color='blue', label=my_label_2)

    plt.xlabel(r'$n_E$', fontsize=15)
    plt.ylabel(r'$-\omega_{B}$', fontsize=15)
    plt.xlim([0, 256])
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(which='both', axis='both')
    plt.savefig('fig_vB_coeff.jpg', bbox_inches='tight',
                pad_inches=0.02, dpi=450)
    plt.show()
