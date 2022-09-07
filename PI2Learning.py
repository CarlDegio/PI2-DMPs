import numpy as np
from DMP_System import SystemOptions,DMPOptions,DMPSystem
import functools
import matplotlib.pyplot as plt


class Trajectory:
    """
    可以修改，使其包含多维DMP，还需要对应的生成奖励
    """

    def __init__(self, sys_option: SystemOptions, dmp_option: DMPOptions, n_dmps=1):

        self.dmp_systems = [DMPSystem(sys_option, dmp_option)]

        self.n_bfs = sys_option.nonlinearSys_n_bfs
        self.dt=sys_option.dt
        self.n_dmps = n_dmps

        self.length = int((dmp_option.end_sys_time - dmp_option.start_sys_time) / self.dt)
        self.start_dmp_index = int((dmp_option.start_dmp_time- dmp_option.start_sys_time) / self.dt)
        self.end_dmp_index = int((dmp_option.end_dmp_time - dmp_option.start_sys_time) / self.dt)
        self.y = np.zeros([n_dmps, self.length])  #
        self.yd = np.zeros([n_dmps, self.length])  #
        self.ydd = np.zeros([n_dmps, self.length])  #
        self.t = np.zeros([n_dmps, self.length])  #
        self.x = np.zeros([n_dmps, self.length])  #
        self.mean_canonical = self.dmp_systems[0].nonlinear_sys.basic_fun_mean_canonical  #
        self.weight = np.zeros([n_dmps, self.n_bfs])  # 基础权重，对一个rollout是一样的
        self.eps = np.zeros([n_dmps, self.length, self.n_bfs])  # 实际权重为基础权重+eps
        self.psi = np.zeros([n_dmps, self.length, self.n_bfs])  #
        self.g_term = np.zeros([n_dmps, self.length, self.n_bfs])  #
        self.r_t = np.zeros(self.length)  #
        self.r_end = np.zeros(n_dmps)  #

    def log_step(self, tick):
        for sys_index in range(self.n_dmps):
            self.y[sys_index, tick] = self.dmp_systems[sys_index].y
            self.yd[sys_index, tick] = self.dmp_systems[sys_index].yd
            self.ydd[sys_index, tick] = self.dmp_systems[sys_index].ydd
            self.t[sys_index, tick] = self.dmp_systems[sys_index].t
            self.x[sys_index, tick] = self.dmp_systems[sys_index].nonlinear_sys.x
            self.psi[sys_index, tick] = self.dmp_systems[sys_index].nonlinear_sys.get_psi_now()
            self.g_term[sys_index, tick] = self.dmp_systems[sys_index].calc_g()

    def run_step(self, tick):
        for sys_index in range(self.n_dmps):
            self.dmp_systems[sys_index].run_step(has_dmp=(self.start_dmp_index <= tick < self.end_dmp_index))

    def calc_cost(self):
        Q = 1000
        R = 1
        for sys_index in range(self.n_dmps):
            for i in range(self.start_dmp_index, self.end_dmp_index):
                Meps = self.g_term[sys_index, i].dot(self.eps[sys_index, i]) * self.g_term[sys_index, i] / (
                        np.linalg.norm(self.g_term[sys_index, i]) ** 2)
                norm_term = 0.5 * R * np.linalg.norm(self.weight + Meps) ** 2
                self.r_t[i] += 0.5 * self.ydd[sys_index, i] ** 2 * Q + norm_term
                if i == 6 and self.y[sys_index, i] != 0.1:
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
    def __init__(self, size=10, n_reuse=0):  # 对于样例任务，n_reuse会导致buffer前几个锁定且更新占优，就会使dtheta变成非零常数
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
    def __init__(self, n_dmps=1, std=20):
        """
        只将噪声添加到强度最大的基函数
        """
        self.system_option = SystemOptions()
        self.dmp_option = DMPOptions()

        self.n_dmps = n_dmps  #
        self.n_bfs = self.system_option.nonlinearSys_n_bfs
        self.start_pos = self.system_option.start
        self.goal_pos = self.system_option.goal
        self.start_time = self.dmp_option.start_sys_time
        self.end_time = self.dmp_option.end_sys_time
        self.start_dmp_time = self.dmp_option.start_dmp_time
        self.end_dmp_time = self.dmp_option.end_dmp_time
        self.dt = self.system_option.dt
        self.start_dmp_index = int((self.start_dmp_time - self.start_time) / self.dt)
        self.end_dmp_index = int((self.end_dmp_time - self.start_time) / self.dt)
        self.std = std
        self.repetitions = 10
        self.updates = 300
        self.n_reuse = 0

        self.weight = np.zeros([n_dmps, self.n_bfs])
        self.length = int((self.end_time - self.start_time) / self.dt)
        self.dmp_length = int((self.end_dmp_time - self.start_dmp_time) / self.dt)  # 仿真离散长度
        self.R = np.zeros([self.dmp_length, self.repetitions])
        self.S = np.zeros([self.dmp_length, self.repetitions])
        self.P = np.zeros([self.dmp_length, self.repetitions])
        self.buffer = ReplayBuffer(size=self.repetitions, n_reuse=self.n_reuse)

        self.mean_tune = np.zeros(self.n_bfs)
        self.last_dtheta=np.zeros(self.n_bfs)

    def run(self):
        for i in range(self.updates):
            if i % 10 == 0:
                traj_eval = self.rollout(0)
                print('eval_result', traj_eval.r_t.sum() + traj_eval.r_end)
            if i == self.updates - 1:
                traj_eval = self.rollout(0)
                print(self.weight)
                print(traj_eval.y[0, 30])
                plt.plot(traj_eval.t[0], traj_eval.y[0])
                plt.show()
            noise_gain = max((self.updates - i) / self.updates, 0.1)

            while 1:
                #mean_tune=np.array(input().split(' ')) # 人为修改噪声均值，使权值变化更快
                mean_tune=np.array([0,0,0,0,0,0,0,0,0,0])
                self.mean_tune=[]
                for num in mean_tune:
                    self.mean_tune.append(float(num))
                flag = self.buffer.append(self.rollout(noise_gain))
                if flag:
                    break

            self.pi2_update(10)
            self.buffer.sort()
            self.buffer.pop()
        np.savetxt("weight.txt", self.weight.reshape(-1, 1))

    def rollout(self, noise_gain):
        std_eps = noise_gain * self.std
        traj = Trajectory(self.system_option,self.dmp_option,n_dmps=self.n_dmps)
        last_index = -1  # time是同时的，所以会同时切换
        EPS = np.zeros([self.n_dmps, self.n_bfs])
        # print('weight_eps:',end='')
        for t in range(self.length):
            traj.log_step(t)
            index = traj.psi[0, t].argmax()
            if index != last_index:  # 切换了activate
                EPS = np.zeros([self.n_dmps, self.n_bfs])
                last_index = index
                for sys_index in range(self.n_dmps):
                    eps = np.random.normal(loc=self.mean_tune[index], scale=1.0)  # 仅扰动当前时间activate的base function
                    # print(eps,end=' ')
                    eps=eps*std_eps
                    EPS[sys_index, index] = eps
                    traj.dmp_systems[sys_index].set_weight(self.weight + EPS[sys_index])
                traj.eps[:, t, :] = EPS
            else:
                traj.eps[:, t, :] = EPS
            traj.run_step(t)
        traj.calc_cost()
        # print()
        # print(f'train_result_{len(self.buffer)}', traj.r_t.sum() + traj.r_end)
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
        self.weight += 0.99*self.last_dtheta+dtheta # Momentum，使权值大移动时更快
        self.last_dtheta=dtheta


if __name__ == "__main__":
    learn = PI2LearningPer(1, 9)
    learn.run()
    plt.show()
