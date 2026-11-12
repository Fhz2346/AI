import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))


def load_map(fname):
  mapdata = np.loadtxt(fname,dtype={
    'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
    '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
    goal = np.atleast_2d(goal)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    hb = draw_block_list(ax, blocks)
    hs = ax.plot(
        start[0:1], start[1:2], start[2:], "ro", markersize=7, markeredgecolor="k"
    )
    hg = ax.plot(
        goal[:, 0], goal[:, 1], goal[:, 2], "go", markersize=7, markeredgecolor="k"
    )
    for i, (x, y, z) in enumerate(goal):
        ax.text(x + 0.7, y + 0.2, z, str(i + 1), fontsize=10, color="k")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(boundary[0, 0], boundary[0, 3])
    ax.set_ylim(boundary[0, 1], boundary[0, 4])
    ax.set_zlim(boundary[0, 2], boundary[0, 5])
    return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
    """
    Subroutine used by draw_map() to display the environment blocks
    """
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype="float",
    )
    f = np.array(
        [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )
    clr = blocks[:, 6:] / 255
    n = blocks.shape[0]
    d = blocks[:, 3:6] - blocks[:, :3]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype="int64")
    fcl = np.zeros((6 * n, 3))
    for k in range(n):
        vl[k * 8 : (k + 1) * 8, :] = v * d[k] + blocks[k, :3]
        fl[k * 6 : (k + 1) * 6, :] = f + k * 8
        fcl[k * 6 : (k + 1) * 6, :] = clr[k, :]

    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors="k")
        pc.set_facecolor(fcl)
        h = ax.add_collection3d(pc)
        return h


def is_time_met(times, stay_time):
    goal_stays = [stay_time.goal_1] + [stay_time.goal_other] * (len(times) - 1)
    goal_event_times = np.cumsum(goal_stays)
    path_event_times = np.cumsum(times)
    return path_event_times <= goal_event_times


def animate_dynamic_target(ax, goal, paths, times, goal1_stay, goal_other_stay, slowdown=5):
  goal_stays = [goal1_stay] + [goal_other_stay] * (goal.shape[0] - 1)
  goal_event_times = np.cumsum(goal_stays)
  path_event_times = np.cumsum(times)

  goal_idx = 0
  path_idx = 0
  current_goal_handle = ax.plot(
      [goal[0,0]], [goal[0,1]], [goal[0,2]],
      'bo', markersize=9, markeredgecolor='k'
  )
  current_path_handle = None
  start_time = time.time()

  while goal_idx < len(goal) or path_idx < len(paths):
    sim_time = (time.time() - start_time) / slowdown

    if goal_idx < len(goal) and sim_time >= goal_event_times[goal_idx]:
      if current_goal_handle is not None:
        current_goal_handle[0].remove()
      if goal_idx < len(goal) - 1:
        current_goal = goal[goal_idx + 1]
        current_goal_handle = ax.plot(
            [current_goal[0]], [current_goal[1]], [current_goal[2]],
            'bo', markersize=9, markeredgecolor='k'
        )
      goal_idx += 1

    if path_idx < len(paths) and sim_time >= path_event_times[path_idx]:
      if current_path_handle is not None:
        current_path_handle[0].remove()
      path = paths[path_idx]
      current_path_handle = ax.plot(
          path[:, 0], path[:, 1], path[:, 2],
          'r-', linewidth=2
      )
      path_idx += 1

    plt.draw()
    plt.pause(0.01)

  if current_path_handle is not None:
    current_path_handle[0].remove()
  return path_event_times <= goal_event_times
