% ----- Initialize ----- %
clear;
clc;
close all;
ComputerName = getenv('computername');
FileName = ['MonteCarloSimulation',ComputerName,date];
% ----- Parameters ----- %
plot_ind = 1;
calc_KL = 0;
fusion_ind = 1;
TotalTimeSteps = 100;
dt = 0.05;
Nf = 10; % number of estimators
Q = [0.001 0 0 0 0;...
    0 0.001 0 0 0;...
    0 0 0.001 0 0;...
    0 0 0 0.02 0;...
    0 0 0 0 0.02]; % process noise covariance matrix

P0 = [1 0 0 0 0;...
    0 1 0 0 0;...
    0 0 1 0 0;...
    0 0 0 3 0;...
    0 0 0 0 3]; % initalization covariance matrix

R1 = [0.002 0 0 0;...
    0 0.002 0 0;...
    0 0 0.002 0;...
    0 0 0 0.002]; % Measuements noise (radius)
R2 = [0.006 0 0 0;...
    0 0.006 0 0;...
    0 0 0.006 0;...
    0 0 0 0.006];% Measuements noise (angle)

X0 = [0,0,0,0.5,1]; % initalization states mean

Na = 11; % number of algorithms
Np =200; % number of particles
Np_c = 500;
Q_theta = Q(3,3)*2;
Q_v = Q(4,4)*2;
Q_theta_dot = Q(5,5)*2;
Q_x = Q(1,1)*2;
Q_y = Q(2,2)*2;
Nmc = 100;
Nm = 1;

% ---- Initialize Veriables ---- %
Z = zeros(Nm,2,Nf);
MSE = zeros(TotalTimeSteps,Nmc,Nf,Na);
Particles = zeros(5,Np,Nf,Na);
Weights = zeros(Np,Nf,Na);
Particles_optimal = zeros(5,Np_c);
Weights_optimal = ones(1,Np_c);



Sensor_pose = [-5,0;0,5;5,0;0, -2];

tic;
for MCi = 1:Nmc
    %     Sensor_pose = unifrnd(-10,10,Nf,2);
    X = mvnrnd(X0,0.1.*P0);
    
    for jj = 1:Nf
        for kk = 1:Na
            Particles(:,:,jj,kk) = mvnrnd(X0,0.1.*P0,Np)';
        end
    end
    Particles_optimal = mvnrnd(X0,0.1.*P0,Np_c)';
    
    for t = 1:TotalTimeSteps
        % ----- process time propogation ----- %
        ProcessNoise = mvnrnd(zeros(1,5),Q);
        X(4:5) = X(4:5) + ProcessNoise(4:5);
        X(3) = X(3) + X(4).*dt + ProcessNoise(3);
        X(2) = X(2) + X(5).*sin(X(3)).*dt + ProcessNoise(2);
        X(1) = X(1) + X(5).*cos(X(3)).*dt + ProcessNoise(1);
        
        % ----- mesurments model ----- %
        for ii = 1:Nf
            Z(:,:,ii) = measure_GMM([X(3),X(1),X(2)],Nm);
        end
        
        
        % ----- Time propogation ----- %
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_v),[1,Np,Nf,Na]);
        Particles(5,:,:,:) = Particles(5,:,:,:) + EstimatorsProcessNoise;
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_theta_dot),[1,Np,Nf,Na]);
        Particles(4,:,:,:) = Particles(4,:,:,:) + EstimatorsProcessNoise;
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_theta),[1,Np,Nf,Na]);
        Particles(3,:,:,:) = Particles(3,:,:,:) + Particles(4,:,:,:).*dt + EstimatorsProcessNoise;
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_y),[1,Np,Nf,Na]);
        Particles(2,:,:,:) = Particles(2,:,:,:) + Particles(5,:,:,:).*sin(Particles(3,:,:,:)).*dt + EstimatorsProcessNoise;
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_x),[1,Np,Nf,Na]);
        Particles(1,:,:,:) = Particles(1,:,:,:) + Particles(5,:,:,:).*cos(Particles(3,:,:,:)).*dt + EstimatorsProcessNoise;
        % ----- Time propogation optimal----- %
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_v),[1,Np_c]);
        Particles_optimal(5,:) = Particles_optimal(5,:) + EstimatorsProcessNoise;
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_theta_dot),[1,Np_c]);
        Particles_optimal(4,:) = Particles_optimal(4,:) + EstimatorsProcessNoise;
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_theta),[1,Np_c]);
        Particles_optimal(3,:) = Particles_optimal(3,:) + Particles_optimal(4,:).*dt + EstimatorsProcessNoise;
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_y),[1,Np_c]);
        Particles_optimal(2,:) = Particles_optimal(2,:) + Particles_optimal(5,:).*sin(Particles_optimal(3,:)).*dt + EstimatorsProcessNoise;
        EstimatorsProcessNoise = normrnd(0,sqrt(Q_x),[1,Np_c]);
        Particles_optimal(1,:) = Particles_optimal(1,:) + Particles_optimal(5,:).*cos(Particles_optimal(3,:)).*dt + EstimatorsProcessNoise;
        
        % ----- Weighting ----- %
        for ii = 1:Na
            for jj = 1:Nf
                for kk = 1:Np
                    Weights(kk,jj,ii) = Likelihood_GMM([Particles(3,kk,jj,ii),Particles(1,kk,jj,ii),Particles(2,kk,jj,ii)],Z(:,:,jj));
                end
                Weights(:,jj,ii) = Weights(:,jj,ii)./sum(Weights(:,jj,ii));
                if sum(isnan(Weights(:,jj,ii)))
                    Weights(:,jj,ii) = ones(1,Np)./Np;
                end
            end
        end
        
        
        
        
        % ----- Weighting optimal ----- %
        
        Z_optimal = [];
        for ii = 1:Nf
            Z_optimal = [Z_optimal;Z(:,:,ii)];
        end
        for kk = 1:Np_c
            Weights_optimal(kk) = Likelihood_GMM([Particles_optimal(3,kk),Particles_optimal(1,kk),Particles_optimal(2,kk)],Z_optimal);
        end
        Weights_optimal = Weights_optimal./sum(Weights_optimal);
        if sum(isnan(Weights_optimal))
            Weights_optimal = ones(1,Np_c)./Np_c;
        end
        
        
        
        % ----- resampling ----- %
        for ii = 1:Na
            for jj = 1:Nf
                Particles(:,:,jj,ii) = Particles(:,(randsample(Np,Np,true, Weights(:,jj,ii))),jj,ii) ;
                Weights(:,jj,ii) = ones(Np,1)./Np;
            end
        end
        Particles_optimal = Particles_optimal(:,(randsample(Np_c,Np_c,true, Weights_optimal))) ;
        Weights_optimal = ones(1,Np_c)./Np_c;
        
        
        % ----- Fusion ----- %
        if fusion_ind == 1
            % --- optimal fusion using Particles Intersection ---- %
            OptimalFusionMethod = ["trace","det","ext_det","MAP","Or","trace","det","ext_det","MAP","Or"];
            for id_method = 1:5
                alpha = optimal_fusion(Particles(1:2,:,:,id_method),Weights(:,:,id_method),OptimalFusionMethod(id_method));
                for jj = 1:Nf
                    Weights_temp(:,jj) = ParticlesIntersection(jj,Particles(:,:,:,id_method),Weights(:,:,id_method),alpha,0.2);
                end
                Weights(:,:,id_method) = Weights_temp;
                for jj = 1:Nf
                    Weights(:,jj,id_method) = Weights(:,jj,id_method)./sum(Weights(:,jj,id_method));
                    Particles(:,:,jj,id_method) = Particles(:,(randsample(Np,Np,true, Weights(:,jj,id_method))),jj,id_method);
                end
            end
            
            % --- optimal fusion using Linear Particles Intersection ---- %
            
            for id_method = 6:10
                alpha = optimal_fusion(Particles(1:2,:,:,id_method),Weights(:,:,id_method),OptimalFusionMethod(id_method));
                Particles_temp = LinearParticlesIntersection(Particles(:,:,:,id_method),Weights(:,:,id_method),alpha,0.001);
                for jj = 1:Nf
                    Weights(:,jj,id_method) = Weights(:,jj,id_method)./sum(Weights(:,jj,id_method));
                    Particles(:,:,jj,id_method) = Particles_temp;
                end
            end
            
        end
        % ----- KL-calculating ----- %
        if calc_KL == 1
            for ii = 1:Nf
                for jj = 1:Na
                    KL(t,MCi,ii,jj) = find_KL(Particles_optimal(1:2,:),Particles(1:2,:,ii,jj),0.005);
                end
            end
        end
        %----- Error calculating ----- %
        for ii = 1:Nf
            for jj = 1:Na
                MSE(t,MCi,ii,jj) = norm(X(1:2)'-mean(Particles(1:2,:,ii,jj),2));
            end
            MSE(t,MCi,ii,jj+1) = norm(X(1:2)'-mean(Particles_optimal(1:2,:),2));
        end
        
        %----- Plootting ----- %
        if plot_ind == 1
            col = [0.2,0.2,0.2;0,0,1;0,1,0;0,1,1;1,0,0;1,0,1;1,1,0;0.5,0.5,1;0.5,0.5,0.5;0.5,0.5,0.5;0.5,0.5,0.3;0.5,0.5,0.1;0.5,0.3,0.4];
            for ii = 1:1
                figure(ii);
                subplot(2,1,1);
                plot(X(1),X(2),'k+','MarkerSize',20);
                hold on;
                for jj = 1:Na
                    plot((Particles(1,:,ii,jj)),(Particles(2,:,ii,jj)),'.','Color',col(jj,:),'MarkerSize',2);
                end
                for jj = 1:Na
                    plot(mean(Particles(1,:,ii,jj)),mean(Particles(2,:,ii,jj)),'*','Color',col(jj,:),'MarkerSize',5);
                end
                xlim([-2 2]);
                ylim([-2 2]);
                hold off;
                axis equal;
                subplot(2,1,2);
                hold on;
                for jj = 1:(Na-1)
                    plot(1:t,MSE(1:t,MCi,ii,jj),'Color',col(jj,:),'MarkerSize',5);
                end
                plot(t,norm(X(1:2)'-mean(Particles(1:2,:,ii,Na),2)),'*','Color',col(Na,:),'MarkerSize',5);
                plot(t,norm(X(1:2)'-mean(Particles_optimal(1:2,:),2)),'+','Color',col(Na,:),'MarkerSize',10);
                legend('PI trace', 'PI det', 'PI extended det','PI MAP','PI Or',...
                    'LPI trace', 'LPI det', 'LPI extended det','LPI MAP','LPI Or','No fusion','Centralized');
                if calc_KL == 1
                    subplot(3,1,3);
                    hold on;
                    for jj = 1:(Na-1)
                        plot(1:t,KL(1:t,MCi,ii,jj),'Color',col(jj,:),'MarkerSize',5);
                    end
                    plot(1:t,KL(1:t,MCi,ii,Na),'*','Color',col(Na,:),'MarkerSize',5);
                    legend('PI trace', 'PI det', 'PI extended det', 'PI MAP',...
                        'LPI trace', 'LPI det', 'LPI extended det', 'LPI MAP','No fusion');
                end
                set(gcf, 'Position',  [10, 10, 500, 1000])
            end
            pause(dt);
        end
        
        
        remining_time = (toc/(t*MCi))*(Nmc*TotalTimeSteps-MCi*t);
        disp(['Monte Carlo iter:',num2str(MCi),', Time step:',num2str(t),', remining time:',num2str(remining_time/60),'[mim]']);
    end
end
if calc_KL == 1
    save(FileName,'MSE','KL');
else
    save(FileName,'MSE');
end

for ii = 1:Nf
    figure(ii);
    hold on;
    for jj = 1:Na
        plot(1:TotalTimeSteps,mean(MSE(:,:,ii,jj),2))
    end
    legend('PI trace', 'PI det', 'PI extended det', 'PI MAP',...
                        'LPI trace', 'LPI det', 'LPI extended det', 'LPI MAP','No fusion');
end

figure(ii+1);
for ii = 1:Nf
    for jj = 1:Na
        S(1) = std(mean(MSE(:,:,ii,jj),2));
        M(1) = mean(mean(MSE(:,:,ii,jj),2));
        bar((9*ii-(9-jj)),M,'FaceColor','none');
        hold on;
        h(jj) =  errorbar((9*ii-(9-jj)),M,S);
    end
    
end
legend([h],{'PI trace', 'PI det', 'PI extended det', 'PI MAP',...
                        'LPI trace', 'LPI det', 'LPI extended det', 'LPI MAP','No fusion'});

markers = ['o','^','s','p','o','^','s','p','.','.','s','p'];
facecolor = ['r','r','r','r','b','b','b','b','w','g','g','g'];
for ii = 1:Nf
    figure(ii+4);
    for jj = 1:Na-1
        S(1) = std(mean(MSE(:,:,ii,jj),2));
        M(1) = mean(mean(MSE(:,:,ii,jj),2));
        bar(jj,M,'FaceColor',facecolor(jj),'EdgeColor','black');
        hold on;
        h(jj) =  errorbar(jj,M,S,'Color','black','Marker',markers(jj),'MarkerSize',8,'MarkerFaceColor','white');
    end
    S(1) = std(mean(MSE(:,:,ii,Na+1),2));
    M(1) = mean(mean(MSE(:,:,ii,Na+1),2));
    bar(jj+1,M,'FaceColor',facecolor(Na+1),'EdgeColor','black');
    h(jj+1) =  errorbar(jj+1,M,S,'Color','black','Marker',markers(Na+1),'MarkerSize',8,'MarkerFaceColor','white');
    
    S(1) = std(mean(MSE(:,:,ii,Na),2));
    M(1) = mean(mean(MSE(:,:,ii,Na),2));
    bar(jj+2,M,'FaceColor',facecolor(Na),'EdgeColor','black');
    hold on;
    h(jj+2) =  errorbar(jj+2,M,S,'Color','black','Marker',markers(Na),'MarkerSize',8,'MarkerFaceColor','white');
    
    ylabel('MSE');
    set(gca,'XTick',[2.5 6.5 9],'XTickLabel',{'PI','LPI','Centralized'});
    legend([h],{'trace', 'det', 'extended det', 'MAP'});
end
if calc_KL == 1
    for ii = 1:Nf
        figure(ii+7);
        for jj = 1:Na
            st = mean(KL(:,:,ii,jj),2);
            st = st(~isinf(st));
            S(1) = std(st);
            M(1) = mean(st);
            bar(jj,M,'FaceColor',facecolor(jj),'EdgeColor','black');
            hold on;
            h(jj) =  errorbar(jj,M,S,'Color','black','Marker',markers(jj),'MarkerSize',8,'MarkerFaceColor','white');
        end
        
        ylabel('KL');
        set(gca,'XTick',[2.5 6.5 9],'XTickLabel',{'PI','LPI','No fusion'});
        legend([h],{'trace', 'det', 'extended det', 'MAP'});
    end
end
