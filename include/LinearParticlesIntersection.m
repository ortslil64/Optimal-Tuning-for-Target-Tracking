function [Particles_out] = LinearParticlesIntersection(Particles,Weights,alpha,beta)
Nf = size(Particles,3);
Np = size(Particles,2);
Nd = size(Particles,1);
alpha = alpha./sum(alpha);
Particles_temp = [];
for ii = 1:Nf
    Particles_temp = [Particles(:,randsample(Np,ceil(Np.*alpha(ii)),true,Weights(:,ii)),ii),Particles_temp];
end

Particles_out = Particles_temp(:,1:Np);
end

