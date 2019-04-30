function [samples] = measure_GMM(x,n)
Nc = 4;
C(:,:,1) = [0.001,0;0,0.003];
C(:,:,2) = [0.001,0;0,0.003];
C(:,:,3) =[0.001,0;0,0.003];
C(:,:,4) =[0.001,0;0,0.003];
M = [0,0.2;0.2,0;-0.2,0;0,-0.2];
R = [cos(x(1)), -sin(x(1));sin(x(1)), cos(x(1))];
Mt = (R*M')';
Mt = Mt + x(2:3);
for ii = 1:Nc
   Ct(:,:,ii) = R*C(:,:,ii)*R';
end
GM = gmdistribution(Mt,Ct);
samples = random(GM,n);

% plot(x(2),x(3),'r.',Mt(:,1),Mt(:,2),'k.');
% hold on;
% plot(samples(:,1),samples(:,2),'r.');
end

