function [alpha] = optimal_fusion(Particles,Weights,string)
Nf = size(Particles,3);
Np = size(Particles,2);
Nd = size(Particles,1);
if strcmp(string,'trace')
    for ii = 1:Nf
        Tc(ii) = 1/trace(cov(Particles(:,:,ii)'));
    end
    for ii = 1:Nf
        alpha(ii) = Tc(ii)./sum(Tc);
    end
end

if strcmp(string,'det')
    for ii = 1:Nf
        Tc(ii) = 1/det(cov(Particles(:,:,ii)'));
    end
    for ii =1:Nf
        alpha(ii) = Tc(ii)./sum(Tc);
    end
end

if strcmp(string,'ext_det')
    for ii = 1:Nf
        Tc(:,:,ii) = inv(cov(Particles(:,:,ii)'));
    end
    P = sum(Tc,3);
    Ts = 0;
    for ii = 1:Nf
        Ts = Ts + det(Tc(:,:,ii)) - det(P- Tc(:,:,ii));
    end
    for ii =1:Nf
        alpha(ii) = (det(P)+det(Tc(:,:,ii))-det(P-Tc(:,:,ii)))/(3*det(P) + Ts);
    end
    if sum(alpha)==inf || sum(alpha)==-inf 
        alpha = ones(Nf)/Nf;
    end
end

if strcmp(string,'Chernoff')
    warning('off','all');
    options = optimoptions('fmincon','Algorithm','interior-point','MaxIterations',100,'StepTolerance',1e-4,'Display','off');
    beta = 0.005;
    K1 = 3;
    K2 = 5000;
    Chernoff_optim = @(x) -log(IS_ChernoffInformation(Particles,Weights,x,beta,K1,K2));
    Chernoff_optim2 = @(x) -log(IN_ChernoffInformation(Particles,Weights,x,beta));
    x0 = ones(Nf,1)./Nf;
    A0 = eye(Nf);
    b0 = ones(Nf,1);
    Aeq = ones(1,Nf);
    beq = 1;
    lb = zeros(Nf,1);
    ub = ones(Nf,1);
    [alpha,~] = fmincon(Chernoff_optim,x0,A0,b0,Aeq,beq,lb,ub,[],options);
    %     disp(['Optimal a by Chernoff:',mat2str(alpha,3)]);
end

if strcmp(string,'MAP')
    Nd = size(Particles,1);
    alpha_old = unifrnd(0,1,Nf,1);
    alpha_old = alpha_old/sum(alpha_old);
    beta = 0.005;
    GM = cell(Nf,1);
    for kk = 1:Np
        sigma(:,:,kk)=beta*eye(Nd);
    end
    e = 1;
    for ii = 1:Nf
        GM{ii} = gmdistribution(Particles(:,:,ii)',sigma,Weights(:,ii)');
    end
    while e > 0.01
        Q = LinearParticlesIntersection(Particles(:,:,:),Weights(:,:),alpha_old,0.01);
        
        GM_Q = gmdistribution(Q',sigma);
        for ii = 1:Nf
            a(ii) = sum(pdf(GM{ii},Q'));
        end
        for ii = 1:Nf
            alpha_new(ii) = a(ii)/sum(a);
        end
        e = norm(alpha_old-alpha_new);
        alpha_old = alpha_new;
    end
    alpha = alpha_new;
end

if strcmp(string,'Or')
    Nd = size(Particles,1);
    beta = 0.005;
    GM = cell(Nf,1);
    for kk = 1:Np
        sigma(:,:,kk)=beta*eye(Nd);
    end
    for ii = 1:Nf
        GM{ii} = gmdistribution(Particles(:,:,ii)',sigma,Weights(:,ii)');
    end
    for ii = 1:Nf
        for jj = 1:Nf
            a(ii,jj) = sum(pdf(GM{ii},Particles(:,:,jj)'))/Np;
        end
    end
    a=a./repmat(sum(a,2),1,Nf);
    a = a^10;
    alpha = a(1,:);
end

if sum(isnan(alpha))
    alpha = ones(1,Nf)./Nf;
end
end

