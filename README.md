# PI2-DMPs

使用强化学习的方法PI2更新DMPs以实现轨迹模仿。

dmp.py实现了DMPs中的仿真，并且可以拟合轨迹，给出了运行斜坡信号的例子。

PI2Learning实现了PI2学习与更新，可以按照奖励（cost）规则实现最小化cost的轨迹，示例设计了viapoint的模式，即0.3s时尽可能通过y=0.1。

细节见doc.pdf