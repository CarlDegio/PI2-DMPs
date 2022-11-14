function R=viapoint(D)
% implements a simple squared acceleration cost function with a penalty on
% the length of the parameter vector. Additionally, a via point has to be traversed at
% certain moment of time

global n_rfs;
global n_dmps;
global dcps;

n_reps = length(D);
n      = length(D(1).q);         % the length of a trajectory in time steps
n_real = D(1).duration/D(1).dt;  % the duration of the core trajectory in time steps --
                                 % everything beyond this time belongs to the terminal cost

% compute cost
RR = 1;
QQ = 1000;

R = zeros(n,n_reps);

% the penalty along the trajectory
for k=1:n_reps,
  r  = zeros(n_real,1);
  rt = zeros(n-n_real,1);
  for i=1:n_dmps,
    % compute range component of noise
    basesTbases = sum(D(k).dmp(i).bases(1:n_real,:).^2,2);
    basesTeps   = sum(D(k).dmp(i).bases(1:n_real,:).*...
      (D(k).dmp(i).theta_eps(1:n_real,:)-ones(n_real,1)*dcps(i).w'),2);
    
    eps_range = D(k).dmp(i).bases(1:n_real,:).*(basesTeps./(basesTbases+1.e-10)*ones(1,n_rfs));
    
    % cost during trajectory
    r  = r + ...
        0.5 * QQ * D(k).qdd(1:n_real,i).^2 + ...
        0.5 * RR * sum((ones(n_real,1)*dcps(i).w'+eps_range).^2,2);
    % terminal cost: penalize distance from goal and remaining velocity
    rt = rt + ...
        1000 * D(k).qd(n_real+1:end,i).^2 + ...
        1000 * (D(k).q(n_real+1:end,i)-D(k).goal(i)).^2;
      
  end
  
  % pass through the point [0.5 0.2] at t=0.25 seconds
  mm = round(0.3/D(1).dt);
  r(mm)  = r(mm) + 10000000000*((D(k).q(mm,1)-0.5)^2 + (D(k).q(mm,2)-0.2)^2);
  
  R(:,k) = [r;rt];
end

