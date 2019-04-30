function [KL] = find_KL(f1,f2,beta)
Np = size(f1,2);
Nd = size(f1,1);


Pt = [f1';f2'];
xmin = min(Pt(:,1))-1;
xmax = max(Pt(:,1))+1;
ymin = min(Pt(:,2))-1;
ymax = max(Pt(:,2))+1;
func = @(x,y) sample_KL(f1,f2,beta,x,y);
KL = integral2(func,xmin,xmax,ymin,ymax,'AbsTol',1e-5,'RelTol',1e-5 );
end
