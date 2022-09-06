import numpy as np
import matplotlib.pyplot as plt


class SystemOptions:
    def __init__(self):
        self.time_zoom = 1  # 时间缩放
        self.dt = 0.05

        self.start = 0
        self.goal = 1
        self.scale_zoom = self.goal - self.start  # G-y0项,用于非线性项的缩放
        self.originSys_alpha_y = 25
        self.originSys_beta_y = 6

        self.nonlinearSys_n_bfs = 10
        self.nonlinearSys_alpha_x = 8


class OriginSystem:
    """
    ydd=alpha_y*(beta_y(goal-y)-yd)
    yd+=tau*ydd*dt
    y+=tau*yd*dt
    tau>1代表加速仿真，tau<1代表减速仿真
    """

    def __init__(self, sys_option: SystemOptions):
        """
        可以用状态空间方程配置极点得到合理的alpha_y,beta_y
        """
        self.tau = sys_option.time_zoom
        self.dt = sys_option.dt
        self.alpha_y = sys_option.originSys_alpha_y
        self.beta_y = sys_option.originSys_beta_y

        self.ydd = 0
        self.yd = 0
        self.y = sys_option.start
        self.goal = sys_option.goal

    def prepare_step(self, nonlinear_term):
        """
        计算加速度
        """
        self.ydd = self.alpha_y * (self.beta_y * (self.goal - self.y) - self.yd) + nonlinear_term

    def run_step(self):
        """
        加速度积分速度，速度积分位置。tau总是和dt一同出现
        """
        self.y += self.yd * self.tau * self.dt
        self.yd += self.ydd * self.tau * self.dt

    def test(self):
        """
        在不同的时间尺度下，因为使用了缩放，所以其收敛模型几乎是一样的（50%时长左右收敛）
        tau*dt表示系统在0~1系统下的虚拟dt。所以总时长不影响dt，只影响仿真次数与tau
        """
        t = np.linspace(self.dt, 3, 30)  # 0~3s的仿真
        self.tau = 1 / 3  # 因为改变时间而需要改变tau
        y_log = []
        for i in range(30):
            y_log.append(self.y)
            self.prepare_step(0)
            self.run_step()
        plt.plot(t, y_log)
        plt.show()


class NonlinearSystem:
    """
    phi_i(x)=exp(-0.5*(x-c_i)**2*D)
    xd=-alpha_x*x (canonical dynamical system)
    x+=tau*xd*dt
    """

    def __init__(self, sys_option: SystemOptions):
        self.tau = sys_option.time_zoom
        self.dt = sys_option.dt
        self.n_bfs = sys_option.nonlinearSys_n_bfs
        self.alpha_x = sys_option.nonlinearSys_alpha_x
        self.scale_zoom = sys_option.scale_zoom

        self.basic_fun_mean_time = np.linspace(0, 1, self.n_bfs + 2)[1:-1]
        self.basic_fun_mean_canonical = np.exp(-self.alpha_x * self.basic_fun_mean_time)  # 正态非线性的均值
        self.basic_fun_weight = np.random.random(self.n_bfs)
        self.sx2 = np.ones(self.n_bfs)  # 用于模仿学习
        self.sxtd = np.ones(self.n_bfs)
        self.basic_fun_var = (np.diff(self.basic_fun_mean_canonical) * 0.55) ** 2
        self.basic_fun_var = 1 / np.hstack((self.basic_fun_var, self.basic_fun_var[-1]))  # 基函数的方差，控制分布覆盖作用范围

        self.x = 1  # 起点为1
        self.xd = 0

    def set_basic_fun_weight(self, weight):
        self.basic_fun_weight = weight

    def get_basic_fun_weight(self):
        return self.basic_fun_weight

    def prepare_step(self):
        self.xd = -self.alpha_x * self.x

    def run_step(self):
        self.x += self.xd * self.tau * self.dt

    def get_psi_now(self):
        psi = np.exp(-0.5 * (self.x - self.basic_fun_mean_canonical) ** 2 * self.basic_fun_var)
        return psi

    def calc_nonlinear_term(self):
        f = self.basic_fun_weight.dot(self.calc_g())
        return f

    def calc_g(self):
        psi = self.get_psi_now()
        g = psi / np.sum(psi + 1e-10) * self.x * self.scale_zoom
        return g

    def test(self):  # 试验性画出基函数分布，观察其在时间的分布
        """
        按0~1制定收敛变量x及基函数分布中心
        按真实的start_time等缩放收敛变量x
        就可以让基函数分布在真实时间上
        修改start_time等，不改变dt，只改变仿真的步数和tau
        tau*dt=dt'，dt'可以看作0~1系统上的Δt，增加仿真步数相当于缩小了0~1内的Δt
        """
        start_time = 0
        end_time = 10
        self.tau = 1 / 10
        t = np.linspace(start_time, end_time, int((end_time - start_time) // self.dt))
        y = []
        for i in range(int((end_time - start_time) // self.dt)):
            y.append(self.get_psi_now())
            self.prepare_step()
            self.run_step()
        plt.plot(t, y)
        plt.show()


class DMPOptions:
    def __init__(self):
        self.start_sys_time = 0.0
        self.end_sys_time = 1.0
        self.start_dmp_time = 0.0
        self.end_dmp_time = 1.0


class DMPSystem:
    """
    dmp_time范围内，f_term等于calc_f_term
    超出范围,f_term等于0
    """

    def __init__(self, sys_option: SystemOptions, dmp_option: DMPOptions):
        self.dt = sys_option.dt
        self.sys_option=sys_option
        self.dmp_option=dmp_option
        sys_option.tau = 1 / (dmp_option.end_dmp_time - dmp_option.start_dmp_time)
        self.n_step = int((dmp_option.end_sys_time - dmp_option.start_sys_time) // sys_option.dt)
        self.origin_sys = OriginSystem(sys_option)
        self.nonlinear_sys = NonlinearSystem(sys_option)

        self.t = 0
        self.y = 0
        self.yd = 0
        self.ydd = 0
        self.x = 1
        self.xd = 0
        self.psi = np.zeros(sys_option.nonlinearSys_n_bfs)  # 瞬时的基元值
        self.basic_fun_weight = np.zeros(sys_option.nonlinearSys_n_bfs)  # 权重
        self.nonlinear_term = 0  # nonlinear_term的值

    def run_step(self, has_dmp):
        if has_dmp:
            self.nonlinear_term = self.nonlinear_sys.calc_nonlinear_term()
            self.nonlinear_sys.prepare_step()
            self.nonlinear_sys.run_step()
            self.origin_sys.prepare_step(self.nonlinear_term)
            self.origin_sys.run_step()
        else:
            self.nonlinear_term = 0.0
            self.origin_sys.prepare_step(self.nonlinear_term)
            self.origin_sys.run_step()
        self.y = self.origin_sys.y
        self.yd = self.origin_sys.yd
        self.ydd = self.origin_sys.ydd
        self.x = self.nonlinear_sys.x
        self.xd = self.nonlinear_sys.xd
        self.psi = self.nonlinear_sys.get_psi_now()
        self.basic_fun_weight = self.nonlinear_sys.get_basic_fun_weight()
        self.t += self.dt

    def run_trajectory(self):
        y = np.zeros(self.n_step)
        t = np.zeros(self.n_step)
        for i in range(self.n_step):
            if self.dmp_option.start_dmp_time <= self.t <= self.dmp_option.end_dmp_time:
                self.run_step(has_dmp=True)
            else:
                self.run_step(has_dmp=False)
            y[i] = self.y
            t[i] = self.t
        return y, t

    def run_fit_trajectory(self, target: np.ndarray, target_d: np.ndarray, target_dd: np.ndarray):
        """
        target行向量
        在给定时间的情况下模仿轨迹
        隐含dt匹配
        """
        y0 = target[0]
        g = target[-1]

        X = np.zeros(target.shape)
        G = np.zeros(target.shape)
        x = 1

        for i in range(len(target)):
            X[i] = x
            G[i] = g
            xd = -self.nonlinear_sys.alpha_x * x
            x += xd * self.sys_option.time_zoom * self.dt

        self.origin_sys.scale_zoom = g - y0
        F_target = (target_dd / (self.sys_option.time_zoom ** 2) - self.origin_sys.alpha_y * (
                self.origin_sys.beta_y * (G - target) - target_d / self.sys_option.time_zoom))
        PSI = np.exp(
            -0.5 * ((X.reshape((-1, 1)).repeat(self.nonlinear_sys.n_bfs,
                                               axis=1) - self.nonlinear_sys.basic_fun_mean_canonical.reshape(1, -1)
                     .repeat(target.shape, axis=0)) ** 2) * (
                self.nonlinear_sys.basic_fun_var.reshape(1, -1).repeat(target.shape, axis=0)))
        X *= self.origin_sys.scale_zoom
        self.nonlinear_sys.sx2 = ((X * X).reshape((-1, 1)).repeat(self.nonlinear_sys.n_bfs, axis=1) * PSI).sum(axis=0)
        self.nonlinear_sys.sxtd = ((X * F_target).reshape((-1, 1)).repeat(self.nonlinear_sys.n_bfs, axis=1) * PSI).sum(axis=0)
        self.nonlinear_sys.basic_fun_weight = self.nonlinear_sys.sxtd / (self.nonlinear_sys.sx2 + 1e-10)
        self.set_weight(self.nonlinear_sys.basic_fun_weight)

    def calc_g(self):
        return self.nonlinear_sys.calc_g()

    def set_weight(self, weight):
        self.basic_fun_weight = weight
        self.nonlinear_sys.set_basic_fun_weight(weight)

    def test1(self):
        """
        测试运行状况
        """
        self.set_weight(1000 * np.random.random(self.basic_fun_weight.shape))
        y, t = self.run_trajectory()
        plt.plot(t, y)
        plt.show()

    def test2(self):
        """
        测试模仿轨迹,需要将时间设置到0~1
        """
        target_trajectory = np.linspace(0, 1, self.n_step)
        target_trajectory_d = np.linspace(0.2, 0.2, self.n_step)  # 注意速度和时间，值的配合关系
        target_trajectory_dd = np.linspace(0, 0, self.n_step)
        self.run_fit_trajectory(target_trajectory, target_trajectory_d, target_trajectory_dd)
        y, t = self.run_trajectory()
        plt.plot(t, y)
        plt.plot(t, target_trajectory, '-')
        plt.legend(['DMPs', 'desire'])
        plt.show()


if __name__ == '__main__':
    system_option = SystemOptions()
    dmp_option=DMPOptions()

    # origin_term=OriginSystem(system_option)
    # origin_term.test()

    # nonlinear_term = NonlinearSystem(system_option)
    # nonlinear_term.test()

    # dmps=DMPSystem(system_option,dmp_option)
    # dmps.test1()

    # dmps = DMPSystem(system_option,dmp_option)
    # dmps.test2()
