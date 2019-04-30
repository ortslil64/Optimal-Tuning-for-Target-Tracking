function [W] = Likelihood_GMM(x,samples)
Nc = 4;
[Ns,~] = size(samples);
C(:,:,1) = [0.001,0;0,0.003];
C(:,:,2) = [0.001,0;0,0.003];
C(:,:,3) =[0.001,0;0,0.003];
C(:,:,4) =[0.001,0;0,0.003];
% M = [0,0.01;0.01,0;-0.01,0;0,-0.01];
M = [0,0.2;0.2,0;-0.2,0;0,-0.2];
R = [cos(x(1)), -sin(x(1));sin(x(1)), cos(x(1))];
Mt = (R*M')';
Mt = Mt + x(2:3);
for ii = 1:Nc
   Ct(:,:,ii) = R*C(:,:,ii)*R';
end
GM = gmdistribution(Mt,1.*Ct);
W = prod((pdf(GM,samples)));

end