import numpy as np
from dmp import DMPs
import functools
import matplotlib.pyplot as plt


class Trajectory:
    """
    可以修改，使其包含多维DMP，还需要对应的生成奖励
    """

    def __init__(self, n_dmps=1, start_time=0.0, end_time=2.0, start_dmp_time=0.0, end_dmp_time=1.0, n_bfs=10, tau=1,
                 dt=0.01):
        self.dmps_systems = [DMPs(start_time=start_time, end_time=end_time, start_dmp_time=start_dmp_time,
                                  end_dmp_time=end_dmp_time, n_bfs=n_bfs, tau=tau)]

        self.n_bfs = n_bfs
        self.length = int((end_time - start_time) / dt)
        self.n_dmps = n_dmps
        self.start_dmp_index = int((start_dmp_time - start_time) / dt)
        self.end_dmp_index = int((end_dmp_time - start_time) / dt)
        self.y = np.zeros([n_dmps, self.length])  #
        self.yd = np.zeros([n_dmps, self.length])  #
        self.ydd = np.zeros([n_dmps, self.length])  #
        self.t = np.zeros([n_dmps, self.length])  #
        self.x = np.zeros([n_dmps, self.length])  #
        self.mean_canonical = self.dmps_systems[0].f_term.mean_canonical  #
        self.weight = np.zeros([n_dmps, n_bfs])  # 基础权重，对一个rollout是一样的
        self.eps = np.zeros([n_dmps, self.length, n_bfs])  # 实际权重为基础权重+eps
        self.psi = np.zeros([n_dmps, self.length, n_bfs])  #
        self.g_term = np.zeros([n_dmps, self.length, n_bfs])  #
        self.r_t = np.zeros(self.length)  #
        self.r_end = np.zeros(n_dmps)  #

    def log_step(self, tick):
        for sys_index in range(self.n_dmps):
            self.y[sys_index, tick] = self.dmps_systems[sys_index].y
            self.yd[sys_index, tick] = self.dmps_systems[sys_index].yd
            self.ydd[sys_index, tick] = self.dmps_systems[sys_index].ydd
            self.t[sys_index, tick] = self.dmps_systems[sys_index].t
            self.x[sys_index, tick] = self.dmps_systems[sys_index].f_term.x
            self.psi[sys_index, tick] = self.dmps_systems[sys_index].f_term.get_psi()
            self.g_term[sys_index, tick] = self.dmps_systems[sys_index].calc_g()

    def run_step(self, tick):
        for sys_index in range(self.n_dmps):
            self.dmps_systems[sys_index].run_step(has_dmp=(self.start_dmp_index <= tick < self.end_dmp_index))

    def calc_cost(self):
        Q = 1000
        R = 1
        for sys_index in range(self.n_dmps):
            for i in range(self.start_dmp_index, self.end_dmp_index):
                Meps = self.g_term[sys_index, i].dot(self.eps[sys_index, i]) * self.g_term[sys_index, i] / (
                            np.linalg.norm(self.g_term[sys_index, i]) ** 2)
                norm_term = 0.5 * R * np.linalg.norm(self.weight + Meps) ** 2
                self.r_t[i] += 0.5 * self.ydd[sys_index, i] ** 2 * Q + norm_term
                if i == 30 and self.y[sys_index, i] != 0.1:
                    self.r_t[i] += 1e10 * (self.y[0, i] - 0.1) ** 2
            self.r_end += 0.5 * (self.yd[sys_index, self.end_dmp_index - 1]) ** 2 * 1000 + 0.5 * (
                    self.y[sys_index, self.end_dmp_index - 1] - 1) ** 2 * 1000
        return self.r_t, self.r_end


def cmp(self: Trajectory, other: Trajectory):
    if self.r_t.sum() + self.r_end > other.r_t.sum() + other.r_end:
        return 1
    elif self.r_t.sum() + self.r_end == other.r_t.sum() + other.r_end:
        return 0
    else:
        return -1


class ReplayBuffer:
    def __init__(self, size=10, n_reuse=5):  # 对于样例任务，n_reuse会导致buffer前几个锁定且更新占优，就会使dtheta变成非零常数
        self.buffer = []
        self.size = size
        self.n_reuse = n_reuse

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index) -> Trajectory:
        return self.buffer[index]

    def append(self, traj: Trajectory):
        self.buffer.append(traj)
        if len(self.buffer) >= self.size:
            return 1  # 需要pop
        else:
            return 0

    def pop(self):
        for i in range(self.size - self.n_reuse):
            self.buffer.pop()

    def sort(self):
        self.buffer.sort(key=functools.cmp_to_key(cmp))


class PI2LearningPer:
    def __init__(self, n_dmps=1, num_basic_function=9, start_pos=0, goal_pos=1, start_time=0.0, end_time=2.0,
                 start_dmp_time=0.0, end_dmp_time=1.0, std=20, dt=0.01):
        """
        只将噪声添加到强度最大的基函数
        """
        self.n_dmps = n_dmps  #
        self.n_bfs = num_basic_function
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.start_time = start_time
        self.end_time = end_time
        self.start_dmp_time = start_dmp_time
        self.end_dmp_time = end_dmp_time
        self.start_dmp_index = int((start_dmp_time - start_time) / dt)
        self.end_dmp_index = int((end_dmp_time - start_time) / dt)
        self.std = std
        self.repetitions = 10
        self.updates = 300
        self.n_reuse = 0
        self.dt = dt

        self.weight = np.zeros([n_dmps, self.n_bfs])
        self.length = int((self.end_time - self.start_time) / self.dt)
        self.dmp_length = int((self.end_dmp_time - self.start_dmp_time) / self.dt)  # 仿真离散长度
        self.R = np.zeros([self.dmp_length, self.repetitions])
        self.S = np.zeros([self.dmp_length, self.repetitions])
        self.P = np.zeros([self.dmp_length, self.repetitions])
        self.buffer = ReplayBuffer(size=self.repetitions, n_reuse=self.n_reuse)

    def run(self):
        for i in range(self.updates):
            if i % 10 == 0:
                traj_eval = self.rollout(0)
                print(traj_eval.r_t.sum() + traj_eval.r_end)
            if i == self.updates - 1:
                traj_eval = self.rollout(0)
                print(self.weight)
                print(traj_eval.y[0, 30])
                plt.plot(traj_eval.t[0], traj_eval.y[0])
                plt.show()
            noise_gain = max((self.updates - i) / self.updates, 0.1)

            while 1:
                flag = self.buffer.append(self.rollout(noise_gain))
                if flag:
                    break

            self.pi2_update(10)
            self.buffer.sort()
            self.buffer.pop()

    def rollout(self, noise_gain):
        std_eps = noise_gain * self.std
        traj = Trajectory(n_dmps=self.n_dmps, dt=self.dt, n_bfs=self.n_bfs)
        last_index = -1  # time是同时的，所以会同时切换
        EPS = np.zeros([self.n_dmps, self.n_bfs])
        for t in range(self.length):
            traj.log_step(t)
            index = traj.psi[0, t].argmax()
            if index != last_index:  # 切换了activate
                EPS = np.zeros([self.n_dmps, self.n_bfs])
                last_index = index
                for sys_index in range(self.n_dmps):
                    eps = np.random.normal(loc=0.0, scale=std_eps)  # 仅扰动当前时间activate的base function
                    EPS[sys_index, index] = eps
                    traj.dmps_systems[sys_index].set_weight(self.weight + EPS[sys_index])
                traj.eps[:, t, :] = EPS
            else:
                traj.eps[:, t, :] = EPS
            traj.run_step(t)
        traj.calc_cost()
        return traj

    def pi2_update(self, h=10):
        for m in range(self.repetitions):
            self.R[:, m] = self.buffer[m].r_t[self.start_dmp_index:self.end_dmp_index]
            self.R[:, m][-1] += self.buffer[m].r_end  # 末奖励
        self.S = np.flip(np.flip(self.R).cumsum(0))  # theta+Meps项包含在奖励中
        maxS = self.S.max(1).reshape(-1, 1)
        minS = self.S.min(1).reshape(-1, 1)
        expS = np.exp(-h * (self.S - minS) / (maxS - minS))
        P = expS / expS.sum(1).reshape(-1, 1)
        PMeps = np.zeros([self.n_dmps, self.repetitions, self.dmp_length, self.n_bfs])
        for sys_index in range(self.n_dmps):
            for m in range(self.repetitions):
                traj = self.buffer[m]
                gTeps = (traj.g_term[sys_index, self.start_dmp_index:self.end_dmp_index] * traj.eps[sys_index,
                                                                                           self.start_dmp_index:self.end_dmp_index]).sum(
                    1)
                gTg = (traj.g_term[sys_index, self.start_dmp_index:self.end_dmp_index] ** 2).sum(1)
                PMeps[sys_index, m] = traj.g_term[sys_index, self.start_dmp_index:self.end_dmp_index] * (
                    (P[:, m] * gTeps / (gTg + 1e-10)).reshape(-1, 1))
        dtheta = PMeps.sum(1)
        traj = self.buffer[0]
        N = np.linspace(self.dmp_length, 1, self.dmp_length)
        W = N.reshape(-1, 1) * traj.psi[0, self.start_dmp_index:self.end_dmp_index]
        W = W / W.sum(0)
        for sys_index in range(self.n_dmps):
            dtheta = (W * dtheta[sys_index]).sum(0)
        self.weight += dtheta


if __name__ == "__main__":
    learn = PI2LearningPer(1, 9)
    learn.run()
    plt.show()
