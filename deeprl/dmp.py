import numpy as np

class DMP(object):

    def __init__(self, nbasis, alpha, beta):
        self.nbasis = nbasis
        self.alpha = alpha
        self.beta = beta

    def fit(self, q_im, qd_im, qdd_im, dt, tau=1.0, goal=None, regularizer=0.0):
        """ Fits the weights of the DMPs RBF to the trajectories given.

        Args:
            q_im: Joint positions. One entry per row
            qd_im: Joint velocities. One entry per row
            qdd_im: Joint accelerations. One entry per row
            dt: Time between two states.
            tau: Temporal scaling factor
            goal: End position of the movement. By default, it is the
                the last joint position
            regularizer: Regularizer of the ridge regression. By default,
                no regularization takes place.
        """
        if not (q_im.shape == qd_im.shape == qdd_im.shape):
            raise ValueError('Joint matrices have to be all equal sized!')

        nsteps, dof = q_im.shape

        if goal is None:
            goal = q_im[-1,:]
        elif goal.shape != (dof,):
            raise ValueError('Goal has to be vector of same DOF as joint matrices!')
        else:
            self.goal = goal

        ft = qdd_im / (tau ** 2) - self.alpha * (self.beta * (goal - q_im) - qd_im / tau)
        Psi = self._construct_basis(nsteps, dt, tau)

        # Estimate RBF weights with ridge regression:
        # w = (Psi' * Psi + a * I)^-1 * Psi' * ft  
        self.w = np.linalg.inv(Psi.T.dot(Psi) + regularizer * np.eye(self.nbasis)).dot(Psi.T).dot(ft)

    def _construct_basis(self, nsteps, dt, tau):
        """ Constructs the RBF functions which are used to approximate a trajectory.

        Args:
            nsteps: Number of time steps
            dt: Time between two steps
            tau: Temporal scaling variable        
        """

        alphaz = 3 / (nsteps * dt - dt)
        Ts = nsteps * dt - dt 

        centers = np.zeros(nbasis)
        bandwidths = np.zeros(nbasis)

        # Space the RBF centers in the phase dimension
        for i in range(nbasis):
            centers[i] = np.exp(-alphaz * i / (nbasis-1) * Ts )

        # Compute the bandwidths of each RBF
        for i in range(nbasis - 1):
            bandwidths[i] = 0.5 / (0.65 * ( centers[i+1] - centers[i] ) ** 2)

        # Make the last one equal to the second last
        bandwidths[-1] = bandwidths[-2] 

        Psi = np.zeros([nsteps, nbasis])

        # Phase variable: zd = - tau * alphaz * z
        Z = 1
        for k in range(nsteps):
            for j in range(nbasis): 
                Psi[k,j] = np.exp( - bandwidths[j] * ( Z - centers[j]) ** 2 ) # RBF activation over time    
            Psi[k,:] = Psi[k,:] * Z / np.sum( Psi[k,:] ) # Normalize basis functions and weights by canonical state
            Z = Z - alphaz * Z * tau * dt # Update phase variable Z = Z + Zd * dt

        return Psi

if __name__ == '__main__':
    import scipy.io as sio

    import matplotlib.pyplot as plt

    D = sio.loadmat('../data/joints.mat')

    q_im = D['q_im']
    qd_im = D['qd_im']
    qdd_im = D['qdd_im']

    assert q_im.shape == qd_im.shape == qdd_im.shape

    nsteps, dof = q_im.shape

    # DMP Parameters
    nbasis = 10
    alpha  = 25
    beta   = 6.25

    dt = 0.002
    tau = 1 # Temporal scaling

    # Construct DMP
    dmp = DMP(nbasis, alpha, beta)
    Psi = dmp._construct_basis(nsteps, dt, tau)

    dmp.fit(q_im, qd_im, qdd_im, dt, tau)

    # Look at the basis functions
    plt.figure()
    for i in range(nbasis):
        plt.plot(Psi[:,i])
    
    # Look at the data
    for i in range(dof):
        plt.figure()
        plt.plot(q_im[:,i], label='q{0}'.format(i+1))
        plt.title('Q')
        plt.legend(loc='best')

    plt.show()