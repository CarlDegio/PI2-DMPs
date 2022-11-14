function runPI2Learning(protocol_name)

% This is a simple implementation of Pi2 learning on a 2 DOF Discrete
% Movement Primitive, i.e., an entirely planar implementation. It serves as
% illustration how PI2 learning works. To allow easy changes, cost
% functions and DMP start state and goal state have been kept modular. The
% experimental protocal is read from a protocal text file, <protocol_name>,
% which should be self-explantory.
%
% This work is based on the paper:
% Theodorou E, Buchli J, Schaal S (2010) Reinforcement learning in high dimensional 
% state spaces: A path integral approach. Journal of Machine Learning Research
%
% The simulation control a 2 DOF point mass with the DMPs. This could be easily 
% changed to be a more complex nonlinear system. 
%
% Stefan Schaal, June 2010

% initializes a 2 DOF DMP -- this is dones as two independent DMPs as the
% Matlab DMPs currently do not support multi-DOF DMPs.

global n_dmps;
global n_rfs;

% number of DMPs used
n_dmps = 2;

% number of basis functions per DMP
n_rfs  = 10;

for i=1:n_dmps
  dcp('clear',i);
  dcp('init',i,n_rfs,sprintf('pi2_dmp_%d',i),0);
  dcp('reset_state',i);
  dcp('set_goal',i,1,1);
  dcp('MinJerk',i);
end


% read the protocol file
protocol = readProtocol(protocol_name);

% run all protocol items
for i=1:length(protocol)
  runProtocol(protocol(i));
end

%--------------------------------------------------------------------------
%%
function protocol=readProtocol(protocol_name)
% parses the protocol file <protocol_name> -- look at the examples, the structure of this
% file should be self-explantory.

fp = fopen(protocol_name,'r');
if fp == -1
  error('Cannot open protocol file');
end

% read all lines, discard lines that start with comment sign
protocol = [];
count    = 0;
while 1
  line = fgetl(fp);
  if ~ischar(line)
    break;
  end
  if numel(line) == 0 || line(1) == '%'
    continue;
  end
  d = textscan(line,'%f %f %f %f %f %f %d %s %f %d %d');
  count = count+1;
  % the <protocol> structure stores all important parameters to run PI2 roll outs
  protocol(count).start       = [d{1};d{2}];
  protocol(count).goal        = [d{3};d{4}];
  protocol(count).duration    = d{5};
  protocol(count).std         = d{6};
  protocol(count).reps        = d{7};
  protocol(count).cost        = char(d{8});
  protocol(count).updates     = d{9};
  protocol(count).bases_noise = d{10};
  protocol(count).n_reuse     = d{11};
end
fclose(fp);

%--------------------------------------------------------------------------n
%%
function runProtocol(p)
% runs a particular protocol item, i.e., one line from the protocol specs

global n_dmps;
global n_rfs;
global dcps;

% the integration time step is set fixed to 10 milli seconds
dt = 0.01;

% the variables of the simulated point mass
q    = zeros(2,1);
qd   = zeros(2,1);
u    = zeros(2,1);

% create the big data matrix for the roll-out data: We simply store all roll-out data in
% this matrix and evalute it late efficiently in vectorized from for learning updates
n = p.duration/dt;
D.dmp(1:n_dmps) = struct(...
  'y',zeros(n,1),...              % DMP pos
  'yd',zeros(n,1),...             % DMP vel
  'ydd',zeros(n,1),...            % DMP acc
  'bases',zeros(n,n_rfs),...      % DMP bases function vector
  'theta_eps',zeros(n,n_rfs),...  % DMP noisy parameters
  'psi',zeros(1,n_rfs));          % DMP Gaussian kernel weights

D.q             = zeros(n,n_dmps);% point mass pos
D.qd            = zeros(n,n_dmps);% point mass vel
D.qdd           = zeros(n,n_dmps);% point mass acc
D.u             = zeros(n,n_dmps);% point mass command
D.duration      = p.duration;     % nominal DMP duration
D.dt            = dt;             % time step used
D.goal          = p.goal;         % the goal of the movement

D_eval         = D;  % used for noiseless cost assessment
p_eval         = p;  % used for noiseless cost assessment
p_eval.reps    = 1;
p_eval.std     = 0;
p_eval.n_reuse = 0;

D(1:p.reps) = D;  % one data structure for each repetition

T = zeros(p.updates,2); % used to store the learning trace

for i=1:p.updates
  
  % perform one noiseless evaluation to get the cost
  D_eval=run_rollouts(D_eval,p_eval,1);
 
  % compute all costs in batch from, as this is faster in matlab
  eval(sprintf('R_eval=%s(D_eval);',p_eval.cost));
  if mod(i,10)== 1
    fprintf('%5d.Cost = %f\n',i,sum(R_eval));
  end
  
  T(i,:) = [i*(p.reps-p.n_reuse)+p.n_reuse,sum(R_eval)];
  
  % run learning roll-outs with a noise annealing multiplier
  noise_mult = double(p.updates - i)/double(p.updates);
  noise_mult = max([0.1 noise_mult]);
  D=run_rollouts(D,p,noise_mult);
  
  % compute all costs in batch from, as this is faster vectorized math in matlab
  eval(sprintf('R=%s(D);',p.cost));
  
  % visualization: plot at the start and end of the updating
  if (i==1 || i==p.updates)
    plotGraphs(D,R,p,D_eval,R_eval,p_eval,i)
  end
  
  % perform the PI2 update
  updatePI2(D,R);
  
  % reuse of roll-out: the the n_reuse best trials and re-evalute them the next update in
  % the spirit of importance sampling
  if (i > 1 && p.n_reuse > 0)
    [v,inds]=sort(sum(R,1));
    for j=1:p.n_reuse
      Dtemp = D(j);
      D(j) = D(inds(j));
      D(inds(j)) = Dtemp;
    end
  end
  
end

% perform the final noiseless evaluation to get the final cost
D_eval=run_rollouts(D_eval,p_eval,1);

% compute all costs in batch from, as this is faster in matlab
eval(sprintf('R_eval=%s(D_eval);',p_eval.cost));
fprintf('%5d.Cost = %f\n',i,sum(R_eval));

% plot learning curve
figure(2);
plot(T(:,1),T(:,2));
xlabel('Number of roll outs');
ylabel('Cost');

% plot 2D graph of point mass
figure (3)
plot(D_eval.q(:,1),D_eval.q(:,2));
xlabel('q_1')
ylabel('q_2');
title('2D path of point mass');


%-------------------------------------------------------------------------------
%%
function updatePI2(D,R)
% D is the data structure of all roll outs, and R the cost matrix for these roll outs

global n_dmps;
global n_rfs;
global dcps;

% computes the parameter update with PI2
[n,n_reps] = size(R);

% compute the accumulate cost
S = rot90(rot90(cumsum(rot90(rot90(R)))));

% compute the exponentiated cost with the special trick to automatically
% adjust the lambda scaling parameter
maxS = max(S,[],2);
minS = min(S,[],2);

h = 10; % this is the scaling parameters in side of the exp() function (see README.pdf)
expS = exp(-h*(S - minS*ones(1,n_reps))./((maxS-minS)*ones(1,n_reps)));

% the probabilty of a trajectory
P = expS./(sum(expS,2)*ones(1,n_reps));

% compute the projected noise term. It is computationally more efficient to break this
% operation into inner product terms. 
PMeps = zeros(n_dmps,n_reps,n,n_rfs);

for j=1:n_dmps,
  for k=1:n_reps,
    
    
    % compute g'*eps in vector form
    gTeps = sum(D(k).dmp(j).bases.*(D(k).dmp(j).theta_eps-ones(n,1)*dcps(j).w'),2);
    
    
    % compute g'g
    gTg = sum(D(k).dmp(j).bases.*D(k).dmp(j).bases,2);
    
    % compute P*M*eps = P*g*g'*eps/(g'g) from previous results
    PMeps(j,k,:,:) = D(k).dmp(j).bases.*((P(:,k).*gTeps./(gTg + 1.e-10))*ones(1,n_rfs));

  end
end

% compute the parameter update per time step
dtheta = squeeze(sum(PMeps,2));
% average updates over time

% the time weighting matrix (note that this done based on the true duration of the
% movement, while the movement "recording" is done beyond D.duration). Empirically, this
% weighting accelerates learning
m = D(1).duration/D(1).dt;
N = (m:-1:1)';
N = [N; ones(n-m,1)];

% the final weighting vector takes the kernel activation into account
W = (N*ones(1,n_rfs)).*D(1).dmp(1).psi;

% ... and normalize through time
W = W./(ones(n,1)*sum(W,1));

% compute the final parameter update for each DMP
dtheta = squeeze(sum(dtheta.*repmat(reshape(W,[1,n,n_rfs]),[n_dmps 1 1]),2));

% and update the parameters by directly accessing the dcps data structure
for i=1:n_dmps
  dcps(i).w = dcps(i).w + dtheta(i,:)';
end

%-------------------------------------------------------------------------------
%%
function D=run_rollouts(D,p,noise_mult)
% a dedicated function to run muultiple roll-outs using the specifictions in D and p
% noise_mult allows decreasing the noise with the number of roll-outs, which gives
% smoother converged performance (but it is not needed for convergence).

global n_dmps;
global n_rfs;
global dcps;

dt = D.dt;

% the simulated point mass
mass = 1;
damp = 1;
q    = zeros(2,1);
qd   = zeros(2,1);
u    = zeros(2,1);

% run roll-outs
start = p.n_reuse + 1;
if (D(1).dmp(1).psi(1,1) == 0) % indicates very first batch of rollouts
  start = 1;
end

for k=start:p.reps
  
  % reset the DMP
  for j=1:n_dmps,
    dcp('reset_state',j,p.start(j));
    dcp('set_goal',j,p.goal(j),1);
    q(j) = p.start(j);
    qd(j) = 0;
  end
  
  % integrate for twice the duration to see converence behavior
  for n=1:2*p.duration/dt

    std_eps = p.std * noise_mult;
        
    for j=1:n_dmps
      
      % generate noise
      if ~p.bases_noise % this case adds noise at every time step
        epsilon = randn(n_rfs,1)*std_eps;
      
      else % this case only adds noise for the most active basis function, and noise
           % is not change during the activity of the basis function
        if (n==1)
          epsilon = [randn*std_eps ; zeros(n_rfs-1,1)];
        else
          % what is the max activated basis function from the previous time step?
          [val,ind_basis] = max(D(k).dmp(j).psi(n-1,:));

          % what was the noise vector from the previous time step?
          epsilon_prev = D(k).dmp(j).theta_eps(n-1,:)-dcps(j).w';
          % ... and find the index of the basis function to which we added the noise
          [val,ind_eps] = max(abs(epsilon_prev));
          
          % only add new noise if max basis function index changed
          if (ind_eps ~= ind_basis)
            epsilon = zeros(n_rfs,1);
            epsilon(ind_basis) = randn*std_eps;
          else
            epsilon = epsilon_prev';
          end
          
        end
         
      end
      
      
      % after duration/dt no noise is added anymore
      if n > p.duration/dt
        epsilon = epsilon * 0;
      end
      
      % integrate DMP
      [y,yd,ydd,b]=dcp('run',j,p.duration,dt,0,0,1,1,epsilon);

      % integrate simulated 2D point mass with inverse dynamics control
      % based on DMP output -- essentially, this just perfectly realizes
      % the DMP output, but one could add noise to this equation to make
      % it more interesting, or replace the linear point mass with a nonlinear
      % point mass or other system
      kp = 1;
      kd = 2*sqrt(kp);
      u(j)   = mass * ydd + damp * qd(j) + kp*(y-q(j)) + kd*(yd-qd(j));
      qdd(j) = (u(j) - qd(j) * damp)/mass;
      qd(j)  = qdd(j) * dt + qd(j);
      q(j)   = qd(j) * dt + q(j);
      
      D(k).dmp(j).y(n)   = y;
      D(k).dmp(j).yd(n)  = yd;
      D(k).dmp(j).ydd(n) = ydd;
      D(k).dmp(j).bases(n,:) = b';
      D(k).dmp(j).theta_eps(n,:) = (dcps(j).w+epsilon)';
      D(k).dmp(j).psi(n,:) = dcps(j).psi';
      D(k).q(n,j)   = q(j);
      D(k).qd(n,j)  = qd(j);
      D(k).qdd(n,j) = qdd(j);
      D(k).u(n,j)   = u(j);
      
    end
  end
  
end

%-------------------------------------------------------------------------------
%%
function plotGraphs(D,R,p,D_eval,R_eval,p_eval,figID)
% plots various graphs for one set of roll-outs and the noiseless realization

global n_dmps;
global n_rfs;
global dcps;

gray = [0.5 0.5 0.5];

dt = D.dt;
T  = (1:length(D(1).q))'*dt;

TT  = zeros(length(D(1).q),p.reps);
TTT = zeros(length(D(1).q),p.reps*n_rfs);

figure(figID);
clf;

% pos, vel, acc
for j=1:n_dmps;
  
  % dmp position
  subplot(2*n_dmps,5,(j-1)*10+1);
  
  for k=1:p.reps,
      TT(:,k)=D(k).dmp(j).y;
  end
  plot(T,D_eval(1).dmp(j).y,'Color',gray,'LineWidth',2);
  hold on;
  plot(T,TT);
  hold off;
  ylabel(sprintf('y_%d',j));
  
  title(sprintf('DMP_%d',j));
  
  % dmp velocity
  subplot(2*n_dmps,5,(j-1)*10+2);
    
  for k=1:p.reps,
      TT(:,k)=D(k).dmp(j).yd;
  end
  plot(T,D_eval(1).dmp(j).yd,'Color',gray,'LineWidth',2);
  hold on;
  plot(T,TT);
  hold off;
  ylabel(sprintf('yd_%d',j));
  
  % dmp acceleration
  subplot(2*n_dmps,5,(j-1)*10+3);
  
  for k=1:p.reps,
      TT(:,k)=D(k).dmp(j).ydd;
  end
  plot(T,D_eval(1).dmp(j).ydd,'Color',gray,'LineWidth',2);
  hold on;
  plot(T,TT);
  hold off;
  ylabel(sprintf('ydd_%d',j));
  
  % point mass position
  subplot(2*n_dmps,5,(j-1)*10+4);
  
  for k=1:p.reps,
      TT(:,k)=D(k).q(:,j);
  end
  plot(T,D_eval(1).q(:,j),'Color',gray,'LineWidth',2);
  hold on;
  plot(T,TT);
  hold off;
  ylabel(sprintf('q_%d',j));

  % point mass velocity
  subplot(2*n_dmps,5,(j-1)*10+5);
    
  for k=1:p.reps,
      TT(:,k)=D(k).qd(:,j);
  end
  plot(T,D_eval(1).qd(:,j),'Color',gray,'LineWidth',2);
  hold on;
  plot(T,TT);
  hold off;
  ylabel(sprintf('qd_%d',j));
  
  % the noise profile
  subplot(2*n_dmps,5,(j-1)*10+6);
    
  for k=1:p.reps,
      TTT(:,(k-1)*n_rfs+1:k*n_rfs)=D(k).dmp(j).theta_eps-ones(length(T),1)*dcps(j).w';
  end
  plot(T,TTT);
  ylabel(sprintf('eps_%d',j));

  % the reward (the same for all DMPs)
  subplot(2*n_dmps,5,(j-1)*10+7);

  plot(T,R);
  ylabel(sprintf('reward r'));
  
  % the cumulative reward (the same for all DMPs)
  subplot(2*n_dmps,5,(j-1)*10+7);

  S = rot90(rot90(cumsum(rot90(rot90(R)))));

  plot(T,S);
  ylabel(sprintf('R=sum(r)'));
  
  % the eponentiated and rescaled cumulative reward (the same for all DMPs)
  subplot(2*n_dmps,5,(j-1)*10+8);
  
  maxS = max(S,[],2);
  minS = min(S,[],2);
  
  h = 10;

  expS = exp(-h*(S - minS*ones(1,p.reps))./((maxS-minS)*ones(1,p.reps)));
 
  plot(T,expS);
  ylabel(sprintf('scaled exp(R)'));

  % the paramter vector
  subplot(2*n_dmps,5,(j-1)*10+9);

  bar(dcps(j).w);
  ylabel('theta');
  axis('tight');
 
end

drawnow;






































