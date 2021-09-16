from synthetic_sim import  SpringSim
from customizing import *
import time
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')

args = parser.parse_args()

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    gps_loc_all = list()
    cus_all = list()
    vel_all = list()
    edges_all = list()
    Box_yard = [(52.100093,52.100899),      
         (5.262500, 5.264100)]

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=length,
                                                sample_freq=sample_freq)

        labels = labeling (edges)
        gps_loc = convert_gps(Box_yard,loc)
        cus_data = dataset(labels, gps_loc)
        

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        gps_loc_all.append(gps_loc)
        cus_all.append(cus_data)
        vel_all.append(vel)
        edges_all.append(edges)
    
        

    loc_all = np.stack(loc_all)
    gps_loc_all = np.stack(gps_loc_all)
    cus_all = np.array(cus_all, dtype=object)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, gps_loc_all, cus_all, vel_all, edges_all


print("Generating {} training simulations".format(args.num_train))
loc_train, gps_train, cus_train, vel_train, edges_train = generate_dataset(args.num_train,
                                                     args.length,
                                                     args.sample_freq)


np.save('loc_train' + suffix + '.npy', loc_train)
np.save('gps_train' + suffix + '.npy', gps_train)
np.save('cus_train' + suffix + '.npy', cus_train)
np.save('vel_train' + suffix + '.npy', vel_train)
np.save('edges_train' + suffix + '.npy', edges_train)


print("Generating {} validation simulations".format(args.num_valid))
loc_valid, gps_valid, cus_valid, vel_valid, edges_valid = generate_dataset(args.num_valid,
                                                     args.length,
                                                     args.sample_freq)

np.save('loc_valid' + suffix + '.npy', loc_valid)
np.save('gps_valid' + suffix + '.npy', gps_valid)
np.save('cus_valid' + suffix + '.npy', cus_valid)
np.save('vel_valid' + suffix + '.npy', vel_valid)
np.save('edges_valid' + suffix + '.npy', edges_valid)

print("Generating {} test simulations".format(args.num_test))
loc_test, gps_test, cus_test, vel_test, edges_test = generate_dataset(args.num_test,
                                                  args.length_test,
                                                  args.sample_freq)

np.save('loc_test' + suffix + '.npy', loc_test)
np.save('gps_test' + suffix + '.npy', gps_test)
np.save('cus_test' + suffix + '.npy', cus_test)
np.save('vel_test' + suffix + '.npy', vel_test)
np.save('edges_test' + suffix + '.npy', edges_test)





