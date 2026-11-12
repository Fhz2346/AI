import numpy as np
import time
import matplotlib.pyplot as plt
from run import runtest
from utils import *


def test_single_cube(verbose = True, dynamic_target = False):
  print('Running single cube test...\n') 
  start = np.array([2.3, 2.3, 1.3])
  goal = np.array([7.0, 7.0, 5.5])
  success, pathlength = runtest('./maps/E4_single_cube.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


def test_maze(verbose = True, dynamic_target = False):
  print('Running maze test...\n') 
  start = np.array([0.0, 0.0, 1.0])
  goal = np.array([12.0, 12.0, 5.0])
  success, pathlength = runtest('./maps/E2_maze.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


def test_window(verbose = True, dynamic_target = True):
  print('Running window test...\n') 
  start = np.array([0.2, -4.9, 0.2])
  goal = np.array([
    [8.800, 12.300, 3.800],
    [7.687, 13.227, 4.449],
    [5.000, 13.610, 4.718],
    [2.313, 13.227, 4.449],
    [1.200, 12.300, 3.800],
    [2.313, 11.373, 3.151],
    [5.000, 10.990, 2.882],
    [7.687, 11.373, 3.151],
  ])
  success, pathlength = runtest('./maps/E6_window.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %s'%pathlength)
  print('\n')


def test_tower(verbose = True, dynamic_target = False):
  print('Running tower test...\n') 
  start = np.array([2.5, 4.0, 0.5])
  goal = np.array([4.0, 2.5, 19.5])
  success, pathlength = runtest('./maps/E5_tower.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


def test_flappy_bird(verbose = True, dynamic_target = False):
  print('Running flappy bird test...\n') 
  start = np.array([0.5, 2.5, 5.5])
  goal = np.array([19.0, 2.5, 5.5])
  success, pathlength = runtest('./maps/E1_flappy_bird.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength) 
  print('\n')


def test_room(verbose = True, dynamic_target = True):
  print('Running room test...\n') 
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([[1.7, 0.5 , 1.7],
           [8.0, 1.0, 1.5],
           [6.0, 4.0, 3.0],
           [3.0, 3.6, 0.5],
           [3.0, 7.0 , 1.0],
           [6.0, 8.0, 0.5],
           [8.0, 6.0 , 1.5],
           [9, 7.5, 0.5]])
  success, pathlength = runtest('./maps/E7_room.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %s'%pathlength)
  print('\n')


def test_monza(verbose = True, dynamic_target = False):
  print('Running monza test...\n')
  start = np.array([0.5, 1.0, 4.9])
  goal = np.array([3.8, 1.0, 0.1])
  success, pathlength = runtest('./maps/E3_monza.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


if __name__=="__main__":
    # Static target tests
    # test_single_cube()
    # test_maze()
    # test_flappy_bird()
    # test_monza()
    test_tower()

    # Dynamic target tests
    # test_window()
    # test_room()
    plt.show(block=True)
