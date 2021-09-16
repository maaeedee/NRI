import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random 
from random import choice


vec_prob = [19. / 20,  0, 1. / 20] #spring
vec_type = [0., 0.5, 1.] # spring

# charge_prob=[1. / 2, 0, 1. / 2]
# charge type = [-1., 0., 1.]

class SpringSim(object):
    def __init__(self, n_balls=5, box_size=10000.,  vel_norm=5,
                 interaction_strength=1, noise_var=.0001):
        self.n_balls = n_balls
        self.box_size = box_size
        # self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array(vec_type)
        self._delta_T = 0.001
        self._max_F = 0.5 / self._delta_T # was 0.1

    # def _energy(self, loc, vel, edges):
    #     # disables division by zero warning, since I fix it with fill_diagonal
    #     with np.errstate(divide='ignore'):

    #         K = 0.5 * (vel ** 2).sum()
    #         U = 0
    #         for i in range(loc.shape[1]):
    #             for j in range(loc.shape[1]):
    #                 if i != j:
    #                     r = loc[:, i] - loc[:, j]
    #                     dist = np.sqrt((r ** 2).sum())
    #                     U += 0.5 * self.interaction_strength * edges[
    #                         i, j] * (dist ** 2) / 2
    #         return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])
    

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10,
                          spring_prob=vec_prob):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # # Sample edges
        # edges = np.random.choice(self._spring_types,
        #                          size=(self.n_balls, self.n_balls),
        #                          p=spring_prob)
        # edges = np.tril(edges) + np.tril(edges, -1).T
        # np.fill_diagonal(edges, 0)


        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))

        loc_next = np.zeros((2, n))
        vel_next = np.zeros((2, n))
        

        # Costomizing the sub-groups
        # A = edges
        # G = nx.from_numpy_matrix(A)
        # groups = list(list(G.subgraph(c).nodes) for c in nx.connected_components(G))

        

        ball = self.n_balls
        edges = np.zeros((ball, ball))
        weight_vec = range(1, ball+1)
        max_grp_size = int(0.01*ball)
        min_grp_size = 2
        excpt_list = []
        excpt_w = []
        edges = np.zeros((ball, ball))

        for cntr in range(ball):

            if cntr not in excpt_list: 

                grp_no = choice(range(min_grp_size, max_grp_size))
                if len(excpt_list)<(ball):
                    grp_type = choice([ j for j in weight_vec if (j not in excpt_w)])
                    excpt_w.append(grp_type)
                    for no in range (grp_no):

                        try: 
                        
                            indx = choice([i for i in range(0,ball) if ((i not in excpt_list) & (i != cntr))])
                            excpt_list.append(indx)
                            excpt_list.append(cntr)

                            edges[cntr][indx]=grp_type
                            edges[indx][cntr]=grp_type

                        except: 
                            continue


        all_groups = []
        for i in range(len(weight_vec)):

            temp_mat = (edges==weight_vec[i]).astype(float)
            G = nx.from_numpy_matrix(temp_mat)
            groups = list(list(G.subgraph(c).nodes) for c in nx.connected_components(G))
            # print(groups)

            sub_groups = [j for j in groups if len(j)>1]
            # print('sub_groups: ', sub_groups)
            if len(sub_groups)>=1:

                all_groups.append(sub_groups[0])


        # print('Results: ',all_groups)

        for sub_groups in all_groups: 
            n_sub_groups = len(sub_groups)
            loc_std = random.randint(1,1000)
            # loc_std = [-5, -1, 1, 5, 10]
            # loc_next[:,sub_groups] = np.random.randn(2, n_sub_groups)*loc_std
            loc_next[:,sub_groups] = np.random.uniform(low=0.01, high=1, size=(2, n_sub_groups))*loc_std
            vel_next[:,sub_groups] = np.random.randn(2, n_sub_groups)
            v_norm = np.sqrt((vel_next[:,sub_groups] ** 2).sum(axis=0)).reshape(1, -1)
            vel_next[:,sub_groups] = vel_next[:,sub_groups] * self.vel_norm / v_norm
            loc[0:1, :, sub_groups], vel[0:1, :, sub_groups] = self._clamp(loc_next[:,sub_groups], vel_next[:,sub_groups])


        # loc_next = np.random.randn(2, n) * self.loc_std
        # vel_next = np.random.randn(2, n)
        # v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        # vel_next = vel_next * self.vel_norm / v_norm
        # loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)

            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges


# if __name__ == '__main__':
#     sim = SpringSim()
#     # sim = ChargedParticlesSim()

#     t = time.time()
#     loc, vel, edges = sim.sample_trajectory(T=5000, sample_freq=100)

#     print(edges)
#     print("Simulation time: {}".format(time.time() - t))
#     vel_norm = np.sqrt((vel ** 2).sum(axis=1))
#     plt.figure()
#     axes = plt.gca()
#     axes.set_xlim([-5., 5.])
#     axes.set_ylim([-5., 5.])
#     for i in range(loc.shape[-1]):
#         plt.plot(loc[:, 0, i], loc[:, 1, i])
#         plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
#     plt.figure()
#     # energies = [sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in
#     #             range(loc.shape[0])]
#     plt.plot(energies)
#     plt.show()
