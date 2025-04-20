import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_H(N, K):
    H = np.random.normal(0, 1/np.sqrt(2), size=(N, K)) + \
        1j*np.random.normal(0, 1/np.sqrt(2), size=(N, K))
    return H


def get_tau(N1, N2, K, xi):
    delta_mu = K - N1 - N2
    BB = (xi*(K-N2)+(K-N1))/(delta_mu*xi*K)
    CC = 1/(delta_mu*xi*K)
    tau1 = 0.5*(-BB+np.sqrt((BB**2)-(4*CC)))
    tau2 = 0.5*(-BB-np.sqrt((BB**2)-(4*CC)))
    tau = np.max((tau1, tau2))
    return tau


def get_theta(N, K):
    my_min = np.min((N, K))
    my_max = np.max((N, K))
    my_abs = abs(N-K)
    return my_max*np.log2(my_max)-my_min*np.log2(np.e)-my_abs*np.log2(np.max((my_abs, 1)))


def get_mu(N1, N2, K, xi):
    delta_mu = K-N1-N2
    if delta_mu >= 0:
        mu = K*np.log2(K) + N2*np.log2(xi) - delta_mu*np.log2(np.max((delta_mu, 1))) -\
            (N1+N2)*np.log2(np.e)
    else:
        tau = get_tau(N1, N2, K, xi)
        mu = N1*np.log2(1+K*tau) + N2*np.log2(1+xi*K*tau) - K*np.log2(np.e*tau)

    return mu


def get_RHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B, lambda_A, lambda_B,
            lambda_EA, lambda_EB, rho, psi_A, psi_B):

    data_CB = np.zeros(len(Ps))
    data_CZ = np.zeros(len(Ps))

    for i, P in tqdm(enumerate(Ps)):

        # xi = lambda_B/lambda_EA

        FoT_temp1 = v_A*np.min((n_B, np.max((n_A-n_E, 0))))
        FoT_temp2 = v_B*np.max((n_B-n_E, 0))
        if rho == 1:
            FoT_temp3 = n_A*n_B*np.log2(P)
        else:
            FoT_temp3 = 0
        FoT = np.log2(P)*(FoT_temp1+FoT_temp2)+FoT_temp3

        SoT_temp1 = np.log2(alpha_A)*(np.min((n_B, np.max((n_A-n_E, 0)))))
        SoT_temp2 = np.log2(lambda_B)*np.min((n_B+n_E, n_A)) -\
            np.log2(lambda_EA)*np.min((n_E, n_A))
        SoT_temp3 = get_mu(n_B, n_E, n_A, lambda_B /
                           lambda_EA) - get_theta(n_E, n_A)
        v_A_coeff = v_A*(SoT_temp1-SoT_temp2+SoT_temp3)

        SoT_temp4 = np.log2(alpha_B)*(np.max((n_B-n_E, 0)))
        SoT_temp5 = np.log2(lambda_A)*n_B-np.log2(lambda_EB)*np.min((n_E, n_B))
        SoT_temp6 = get_theta(n_E, n_B)
        v_B_coeff = v_B*(SoT_temp4-SoT_temp5-SoT_temp6)

        if rho == 1:
            kap = n_A*n_B*np.log2((psi_A*psi_B*alpha_A*alpha_B) /
                                  (psi_A*alpha_A*lambda_A+psi_B*alpha_B*lambda_B))
        else:
            kap = 0

        if rho == 1:
            SoT_temp7 = kap - 0
        else:
            SoT_temp7 = kap - np.log2(1-rho**2)

        xak = FoT + v_A_coeff + v_B_coeff + SoT_temp7
        data_CB[i] = xak + v_B*get_theta(n_A, n_B)
        data_CZ[i] = xak + v_B*get_mu(n_A, n_E, n_B, lambda_A/lambda_EB)

    return data_CB, data_CZ


def get_LHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B, lambda_A, lambda_B,
            lambda_EA, lambda_EB, rho, psi_A, psi_B):

    R = 10000
    data_CB = np.zeros(len(Ps))
    data_CZ = np.zeros(len(Ps))

    for i, P in tqdm(enumerate(Ps)):
        ttemp_CB = np.zeros(R)
        ttemp_CZ = np.zeros(R)

        gamma_AB = (alpha_A*P)/lambda_B
        gamma_BA = (alpha_B*P)/lambda_A
        gamma_AE = (alpha_A*P)/lambda_EA
        gamma_BE = (alpha_B*P)/lambda_EB

        if rho == 1:
            kap = n_A*n_B*np.log2((psi_A*psi_B*alpha_A*alpha_B) /
                                  (psi_A*alpha_A*lambda_A+psi_B*alpha_B*lambda_B))
        else:
            kap = 0

        if rho == 1:
            SoT_temp7 = kap - 0
        else:
            SoT_temp7 = kap - np.log2(1-rho**2)

        for j in range(R):

            G_B = get_H(n_E, n_B)
            G_A = get_H(n_E, n_A)
            H_AB = get_H(n_A, n_B)
            H_BA = rho*H_AB.T + np.random.normal(0, np.sqrt((1-rho**2)/2), size=(n_B, n_A)) + \
                1j*np.random.normal(0, np.sqrt((1-rho**2)/2), size=(n_B, n_A))
            H_1 = np.vstack((H_BA, np.sqrt(lambda_B/lambda_EA)*G_A))
            H_2 = np.vstack((H_AB, np.sqrt(lambda_A/lambda_EB)*G_B))

            temp_GB = np.log2(np.linalg.det(
                (gamma_BE*(G_B@G_B.conj().T))+np.eye(n_E)))
            temp_HAB = np.log2(np.linalg.det(
                (gamma_BA*(H_AB@H_AB.conj().T))+np.eye(n_A)))
            temp_H1 = np.log2(np.linalg.det(
                (gamma_AB*(H_1@H_1.conj().T))+np.eye(n_B+n_E)))
            temp_H2 = np.log2(np.linalg.det(
                (gamma_BA*(H_2@H_2.conj().T))+np.eye(n_A+n_E)))
            temp_GA = np.log2(np.linalg.det(
                (gamma_AE*(G_A@G_A.conj().T))+np.eye(n_E)))

            ttemp_CB[j] = v_A*(temp_H1-temp_GA)+v_B*(temp_HAB-temp_GB)
            ttemp_CZ[j] = v_A*(temp_H1-temp_GA)+v_B*(temp_H2-temp_GB)

        data_CB[i] = SoT_temp7 + np.mean(ttemp_CB)
        data_CZ[i] = SoT_temp7 + np.mean(ttemp_CZ)

    return data_CB, data_CZ


def get_range(low=1, high=1e3, count=16):
    temp = np.linspace(np.log10(low), np.log10(high), count)
    return 10**(temp)  # numpy ndarray of shape (count)


# %%
if __name__ == '__main__':

    Ps = get_range()

    n_A = 16
    n_B = 12
    n_E = 10
    v_A = 1
    v_B = 1

    psi_A = 10
    psi_B = 10

    alpha_A = 1.25
    alpha_B = 1.75
    lambda_A = 1.5
    lambda_B = 1.8
    lambda_EA = 0.5
    lambda_EB = 0.25
    rho = 0.5

    LHS_CB, LHS_CZ = get_LHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
                             lambda_A, lambda_B, lambda_EA, lambda_EB, rho, psi_A, psi_B)
    RHS_CB, RHS_CZ = get_RHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
                             lambda_A, lambda_B, lambda_EA, lambda_EB, rho, psi_A, psi_B)

    plt.plot(10*np.log10(Ps), LHS_CB, 'o', color='red',label=r'$n_E={}$'.format(n_E))
    plt.plot(10*np.log10(Ps), RHS_CB, '-', color='blue')
    plt.plot(10*np.log10(Ps), LHS_CZ, 'o', color='red')
    plt.plot(10*np.log10(Ps), RHS_CZ, '--', color='blue')

    n_A = 16
    n_B = 12
    n_E = 14

    LHS_CB, LHS_CZ = get_LHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
                             lambda_A, lambda_B, lambda_EA, lambda_EB, rho, psi_A, psi_B)
    RHS_CB, RHS_CZ = get_RHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
                             lambda_A, lambda_B, lambda_EA, lambda_EB, rho, psi_A, psi_B)

    plt.plot(10*np.log10(Ps), LHS_CB, 'v', color='red',label=r'$n_E={}$'.format(n_E))
    plt.plot(10*np.log10(Ps), RHS_CB, '-', color='green')
    plt.plot(10*np.log10(Ps), LHS_CZ, 'v', color='red')
    plt.plot(10*np.log10(Ps), RHS_CZ, '--', color='green')

    n_A = 16
    n_B = 12
    n_E = 18

    LHS_CB, LHS_CZ = get_LHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
                             lambda_A, lambda_B, lambda_EA, lambda_EB, rho, psi_A, psi_B)
    RHS_CB, RHS_CZ = get_RHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
                             lambda_A, lambda_B, lambda_EA, lambda_EB, rho, psi_A, psi_B)

    plt.plot(10*np.log10(Ps), LHS_CB, '^', color='red',label=r'$n_E={}$'.format(n_E))
    plt.plot(10*np.log10(Ps), RHS_CB, '-', color='orange')
    plt.plot(10*np.log10(Ps), LHS_CZ, '^', color='red')
    plt.plot(10*np.log10(Ps), RHS_CZ, '--', color='orange')

    plt.xlabel('$P$ in dB', fontsize=12)
    plt.ylabel('Upper and Lower Bound', fontsize=9)
    # plt.xlim([0, 256])
    leg = plt.legend(loc='upper left', fontsize=9, frameon=False)
    plt.grid(which='both', axis='both')
    plt.savefig('figfig_simulation.jpg', bbox_inches='tight',
                pad_inches=0.02, dpi=450)
    plt.show()
# %%
if __name__ == '__main__':
    Ps = get_range()

    n_A = 16
    n_B = 12
    n_E = 18
    v_A = 1
    v_B = 1

    psi_A = 10
    psi_B = 10

    alpha_A = 1.25
    alpha_B = 1.75
    lambda_A = 1.5
    lambda_B = 1.8
    lambda_EA = 0.5
    lambda_EB = 0.25
    rho = 0.5

    LHS_CB, LHS_CZ = get_LHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
                             lambda_A, lambda_B, lambda_EA, lambda_EB, rho, psi_A, psi_B)
    RHS_CB, RHS_CZ = get_RHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
                             lambda_A, lambda_B, lambda_EA, lambda_EB, rho, psi_A, psi_B)
    
    ml = r'LB;$\alpha_A={}$,$\alpha_B={}$,$\lambda_A={}$, \n $\lambda_B={}$,$\lambda_EA={}$,$\lambda_EB$'
    mu = r'LB;$\alpha_A={}$,$\alpha_B={}$,$\lambda_A={}$, \n $\lambda_B={}$,$\lambda_EA={}$,$\lambda_EB$'

    plt.plot(10*np.log10(Ps), LHS_CB, 'o', color='red')
    plt.plot(10*np.log10(Ps), RHS_CB, '-', color='blue',
             label=ml.format(alpha_A, alpha_B, lambda_A, lambda_B, lambda_EA, lambda_EB))
    plt.plot(10*np.log10(Ps), LHS_CZ, 'o', color='red')
    plt.plot(10*np.log10(Ps), RHS_CZ, '--', color='blue',
             label=mu.format(alpha_A, alpha_B, lambda_A, lambda_B, lambda_EA, lambda_EB))
    
    plt.xlabel('$P$ in dB', fontsize=12)
    plt.ylabel('Upper and Lower Bound', fontsize=9)
    # plt.xlim([0, 256])
    leg = plt.legend(bbox_to_anchor=(1.04, 0.5),
                     loc='center left', fontsize=9, frameon=False)
    plt.grid(which='both', axis='both')
    plt.savefig('fig_simulation.jpg', bbox_inches='tight',
                pad_inches=0.02, dpi=450)
    plt.show()

    # n_A = 24
    # n_B = 18
    # n_E = 28
    # v_A = 50
    # v_B = 0

    # alpha_A = 1.25
    # alpha_B = 1.75
    # lambda_A = 1.5
    # lambda_B = 1.8
    # lambda_EA = 0.5
    # lambda_EB = 0.25
    # rho = 0.8

    # LHS = get_LHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
    #               lambda_A, lambda_B, lambda_EA, lambda_EB, rho, R)
    # RHS = get_RHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
    #               lambda_A, lambda_B, lambda_EA, lambda_EB, rho)

    # my_label_2 = r'$n_A=$'+str(n_A)+r'$; n_B=$'+str(n_B)+r'$; n_E=$'+str(n_E)+'\n' +\
    #     r'$v_A=$'+str(v_A)+r'$; v_B=$'+str(v_B)+r'$; \rho=$'+str(rho)+'\n'+r'$\alpha_A=$'+str(alpha_A) +\
    #     r'$; \alpha_B=$'+str(alpha_B)+'\n'+r'$\lambda_A=$'+str(lambda_A) +\
    #     r'$; \lambda_B=$'+str(lambda_B)+'\n'+r'$\lambda_{EA}=$'+str(lambda_EA) +\
    #     r'$; \lambda_{EB}=$'+str(lambda_EB)
    # plt.plot(10*np.log10(Ps), LHS, 'o', color='red', label=my_label_2)
    # plt.plot(10*np.log10(Ps), RHS, '-', color='blue')

    # n_A = 24
    # n_B = 18
    # n_E = 28
    # v_A = 50
    # v_B = 0

    # alpha_A = 0.75
    # alpha_B = 0.5
    # lambda_A = 1.5
    # lambda_B = 1.8
    # lambda_EA = 0.25
    # lambda_EB = 0.75
    # rho = 0

    # LHS = get_LHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
    #               lambda_A, lambda_B, lambda_EA, lambda_EB, rho, R)
    # RHS = get_RHS(n_A, n_B, n_E, v_A, v_B, Ps, alpha_A, alpha_B,
    #               lambda_A, lambda_B, lambda_EA, lambda_EB, rho)

    # my_label_2 = r'$n_A=$'+str(n_A)+r'$; n_B=$'+str(n_B)+r'$; n_E=$'+str(n_E)+'\n' +\
    #     r'$v_A=$'+str(v_A)+r'$; v_B=$'+str(v_B)+r'$; \rho=$'+str(rho)+'\n'+r'$\alpha_A=$'+str(alpha_A) +\
    #     r'$; \alpha_B=$'+str(alpha_B)+'\n'+r'$\lambda_A=$'+str(lambda_A) +\
    #     r'$; \lambda_B=$'+str(lambda_B)+'\n'+r'$\lambda_{EA}=$'+str(lambda_EA) +\
    #     r'$; \lambda_{EB}=$'+str(lambda_EB)
    # plt.plot(10*np.log10(Ps), LHS, '^', color='red', label=my_label_2)
    # plt.plot(10*np.log10(Ps), RHS, '-', color='blue')

    # plt.xlabel('$P$ in dB', fontsize=15)
    # plt.ylabel('$C_B$', fontsize=15)
    # # plt.xlim([0, 256])
    # leg = plt.legend(bbox_to_anchor=(1.04, 0.5),
    #                  loc='center left', fontsize=9, frameon=False)
    # plt.grid(which='both', axis='both')

    # mc = ['peachpuff', 'lavender', 'thistle']
    # for i, text in enumerate(leg.get_texts()):
    #     # text.set_fontfamily("Roboto")
    #     # text.set_rotation(20)
    #     # text.set_text(f"Label {i}")
    #     text.set_backgroundcolor(mc[i])
    #     # text.set_fontsize(18)
    #     # text.set_alpha(0.2)
    # plt.savefig('fig_simulation.jpg', bbox_inches='tight',
    #             pad_inches=0.02, dpi=450)
    # plt.show()
