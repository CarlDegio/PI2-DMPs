import numpy as np
import matplotlib.pyplot as plt


class OriginSystemTerm:
    """
    ydd=alpha_y*(beta_y(goal-y)-yd) + force(basic function)
    yd+=tau*ydd
    y+=tau*yd
    tau>1代表加速仿真，tau<1代表减速仿真
    """

    def __init__(self, alpha_y=25, beta_y=6, alpha_g=12, start=0, goal=1, tau=1.0):
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.alpha_g = alpha_g
        self.tau = tau

        self.ydd = 0
        self.yd = 0
        self.y = start

        self.y0 = 0  # 起点
        self.g = goal
        self.dG = 1  # g-y0项,对f项有一个空间缩放

    def prepare_step(self, f):
        self.ydd = self.alpha_y * (self.beta_y * (self.g - self.y) - self.yd) + f

    def run_step(self, dt):
        self.y += self.tau * self.yd * dt
        self.yd += self.tau * self.ydd * dt


class NonlinearTerm:
    """
    phi_i(x)=exp(-0.5*(x-c_i)**2*D)
    xd=-alpha_x*x (canonical dynamical system)

    x+=tau*xd
    """

    def __init__(self, start_time=0.0, end_time=1.0, n_bfs=10, alpha_x=8, tau=1.0):
        self.start = start_time
        self.end = end_time
        self.n_bfs = n_bfs
        self.alpha_x = alpha_x
        self.tau = tau
        self.mean_time = np.linspace(start_time, end_time, n_bfs + 2)[1:-1]  #
        self.mean_canonical = np.exp(-alpha_x / tau * self.mean_time)  # 正态非线性的均值
        self.weight = np.random.random(n_bfs)
        self.sx2 = np.ones(n_bfs)
        self.sxtd = np.ones(n_bfs)
        self.D = (np.diff(self.mean_canonical) * 0.55) ** 2
        self.D = 1 / np.hstack((self.D, self.D[-1]))  # 基函数的方差，控制分布覆盖作用范围

        self.x = 1  # 起点为1
        self.xd = 0

    def set_weight(self, weight):
        self.weight = weight

    def prepare_step(self):
        self.xd = self.tau * (-self.alpha_x * self.x)

    def run_step(self, dt):
        self.x += self.xd * self.tau * dt

    def get_psi(self):
        psi = np.exp(-0.5 * (self.x - self.mean_canonical) ** 2 * self.D)
        return psi

    def calc_f(self, delta_y):
        f = self.weight.dot(self.calc_g(delta_y))
        return f

    def calc_g(self, delta_y):
        psi = self.get_psi()
        g = psi / np.sum(psi + 1e-10) * self.x * delta_y
        return g

    def show(self):  # 试验性画出基函数分布，观察其在时间的分布
        t = np.linspace(self.start, self.end, 100)
        x = np.exp(-self.alpha_x / self.tau * t)
        y = []
        for i in range(self.n_bfs):
            yi = np.exp(-(x - self.mean_canonical[i]) ** 2 / 2 * self.D[i])
            y.append(yi)
        y = np.array(y).T
        plt.plot(t, y)
        plt.show()


class DMPs:
    def __init__(self, start_time=0.0, end_time=2.0, start_dmp_time=0.0, end_dmp_time=1.0, n_bfs=10, tau=1.0, dt=0.01):
        self.tau = tau
        self.start_time = start_time
        self.end_time = end_time
        self.start_dmp_time = start_dmp_time
        self.end_dmp_time = end_dmp_time
        self.sys = OriginSystemTerm(tau=tau)
        self.f_term = NonlinearTerm(start_time=start_dmp_time, end_time=end_dmp_time, tau=tau, n_bfs=n_bfs)
        self.dt = dt
        self.t = 0
        self.y = 0
        self.yd = 0
        self.ydd = 0
        self.x = 0
        self.psi = np.zeros(n_bfs)
        self.weight = np.zeros(n_bfs)
        self.f = 0  # f_term的值

    def run_step(self, has_dmp):
        if has_dmp:
            self.f = self.f_term.calc_f(self.sys.dG)
            self.f_term.prepare_step()
            self.f_term.run_step(self.dt)
            self.sys.prepare_step(self.f)
            self.sys.run_step(self.dt)
        else:
            self.f = 0.0
            self.sys.prepare_step(self.f)
            self.sys.run_step(self.dt)
        self.y = self.sys.y
        self.yd=self.sys.yd
        self.ydd=self.sys.ydd
        self.x = self.f_term.x
        self.psi = self.f_term.get_psi()
        self.weight = self.f_term.weight
        self.t += self.dt

    def run_trajectory(self):
        length = int((self.end_time - self.start_time) / self.dt)
        y = np.zeros(length)
        t = np.zeros(length)
        for i in range(length):
            if self.start_dmp_time <= self.t < self.end_dmp_time:
                self.run_step(has_dmp=True)
            else:
                self.run_step(has_dmp=False)
            y[i] = self.y
            t[i] = self.t
        return y, t

    def run_fit_trajectory(self, target: np.ndarray, target_d: np.ndarray, target_dd: np.ndarray):
        """
        target行向量
        """
        y0 = target[0]
        g = target[-1]

        X = np.zeros(target.shape)
        G = np.zeros(target.shape)
        x = 1

        for i in range(len(target)):
            X[i] = x
            G[i] = g
            xd = -self.f_term.alpha_x * x
            x += xd * self.tau * self.dt

        self.sys.dG = g - y0
        F_target = (target_dd / (self.tau ** 2) - self.sys.alpha_y * (
                self.sys.beta_y * (G - target) - target_d / self.tau))
        PSI = np.exp(
            -0.5 * ((X.reshape((-1, 1)).repeat(self.f_term.n_bfs, axis=1) - self.f_term.mean_canonical.reshape(1, -1)
                     .repeat(target.shape, axis=0)) ** 2) * (self.f_term.D.reshape(1, -1).repeat(target.shape, axis=0)))
        X *= self.sys.dG
        self.f_term.sx2 = ((X * X).reshape((-1, 1)).repeat(self.f_term.n_bfs, axis=1) * PSI).sum(axis=0)
        self.f_term.sxtd = ((X * F_target).reshape((-1, 1)).repeat(self.f_term.n_bfs, axis=1) * PSI).sum(axis=0)
        self.f_term.weight = self.f_term.sxtd / (self.f_term.sx2 + 1e-10)
        self.weight = self.f_term.weight

    def calc_g(self):
        return self.f_term.calc_g(self.sys.dG)

    def set_weight(self, weight):
        self.weight = weight
        self.f_term.set_weight(weight)


if __name__ == "__main__":
    x = np.arange(0, 1, 0.01)  # 对应tau=1下的轨迹
    xd = np.ones(100)
    xdd = np.zeros(100)
    dmps = DMPs(tau=1, n_bfs=10)
    dmps.run_fit_trajectory(x, xd, xdd)
    y, t = dmps.run_trajectory()
    print(dmps.f_term.weight)
    plt.plot(t, y)
    plt.show()
