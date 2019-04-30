function [Weights_out] = ParticlesIntersection(jj,Particles,Weights,alpha,beta)
Nf = size(Particles,3);
Np = size(Particles,2);
Nd = size(Particles,1);
Weights_out = ones(Np,1);
for kk = 1:Np
        sigma(:,:,kk)=beta*eye(Nd);
end
for ii=1:Nf
    GM = gmdistribution(Particles(:,:,ii)',sigma,Weights(:,ii)');
    Weights_out = (pdf(GM,Particles(:,:,jj)').^alpha(ii)).*Weights_out;
end
Weights_out = Weights_out./sum(Weights_out);

if sum(isnan(Weights_out))
    Weights_out = ones(1,Np)./Np;
end
end

