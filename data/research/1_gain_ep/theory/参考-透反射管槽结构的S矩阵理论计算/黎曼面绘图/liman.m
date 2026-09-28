%%
%cr ci黎曼面
%考虑波导n阶CMT
%tube模型
%可以改xj（tube起始位置） wj（tube宽度） hj（tube长度）  
clc;
clear all;
% close all;
f0=3430;
c0=343;
k0=2*pi*f0/c0;
lam=2*pi/k0;
rho = 1.21;
J=3;%单元个数 改的时候要改T_ud_G
j=1:J;%小写L
thetai=45;
thetar=-thetai;
a=abs(lam/(sind(thetar)-sind(thetai)));%确保第一透射分量是传播模式 a周期长度

xunhuan_para1=0;
xunhuan_para12=0;
xunhuan_para2=0;
%EP
h=[0.569 0.195 0.232]*lam;
w=[0.227 0.115 0.153]*a;%波导宽度
d=[0 0.276 0.070 ]*a;%波导硬边界厚度

%DP
% h=[0.211 0.491 0.21]*lam;
% w=[0.137 0.187 0.145]*a;%波导宽度
% d=[0 0.096 0.081 ]*a;%波导硬边界厚度

cii=0.0624-0.05:0.001:0.0624+0.05;
 crr=1.016-0.02:0.001:1.016+0.02;
% cii=0.075:0.0001:0.077;
%  crr=1.0145:0.0001:1.0155;
rzheng_1=zeros(length(crr),length(cii));
rzheng0=zeros(length(crr),length(cii));
rfu0=zeros(length(crr),length(cii));
rfu1=zeros(length(crr),length(cii));
for cr=crr
     xunhuan_para1=xunhuan_para1+1
    for ci=cii
         xunhuan_para12=xunhuan_para12+1;
        xunhuan_para2=xunhuan_para12-(xunhuan_para1-1)*length(cii);
cc=[343 343*(cr+ci*1i) 343];
kc=2*pi*f0./cc;
rhoc=rho*c0./c0*ones(1,J);
% rhoc=rho*c0./cc;
  n=-10:10;%超表面阶数
%  n=-300:300;%超表面阶数
N=length(n);
G=2*pi/a;
an=k0*sind(thetai)+n*G;
an1=k0*sind(-thetai)+n*G;
bn=sqrt(k0^2-an.^2);
bn1=sqrt(k0^2-an1.^2);
% k=0:30;%波导阶数
k=0:10;
K=length(k);
k_j_k_x=((1./w)'*(k)*pi);
% k_j=repmat(k0' , 1, K  );%J*K
k_j_k_z=sqrt((kc'*ones(1,K)).^2-k_j_k_x.^2);
xj=zeros(1,J);
xj(1)=d(1);
for xunhuan1=2:J
xj(xunhuan1)=xj(xunhuan1-1)+w(xunhuan1-1)+d(xunhuan1);
end
tongdao=atan(an./bn)/pi*180;
tongdao1=atan(an1./bn1)/pi*180;
delta_n=[zeros(1,(N-1)/2) 1 zeros(1,(N-1)/2)]';
I=eye(K*J,K*J);%大写i
 U_j_k=zeros(J,K,K);
 for xunhuan1=1:J
for xunhuan2=1:K
U_j_k(xunhuan1,xunhuan2,xunhuan2)=exp(1j*k_j_k_z(xunhuan1,xunhuan2)*2*h(xunhuan1));
end
 end
% U_j_k=  blkdiag(squeeze(U_j_k(1,:,:)),squeeze(U_j_k(2,:,:)),squeeze(U_j_k(3,:,:)),squeeze(U_j_k(4,:,:)),squeeze(U_j_k(5,:,:)),squeeze(U_j_k(6,:,:)));
U_j_k=  blkdiag(squeeze(U_j_k(1,:,:)),squeeze(U_j_k(2,:,:)),squeeze(U_j_k(3,:,:)));

N1=zeros(N,N);
N11=zeros(N,N);
for xunhuan1=1:N
    for xunhuan2=1:N
%           syms x
%         N1(xunhuan1,xunhuan2)=-bn(xunhuan1)./rho./abs(a)*int(exp(1j*(an(xunhuan1)-an(xunhuan2))*x),0,a);
%       f=@(x)    exp(1j*(an(xunhuan1)-an(xunhuan2))*x);
%     N1(xunhuan1,xunhuan2)=-bn(xunhuan1)./rho./abs(a)*integral(f,0,a);
     N1(xunhuan1,xunhuan2)=-bn(xunhuan1)./abs(a)./rho*N1jifen(an(xunhuan1),an(xunhuan2),0,a);
     N11(xunhuan1,xunhuan2)=-bn1(xunhuan1)./abs(a)./rho*N1jifen(an1(xunhuan1),an1(xunhuan2),0,a);
    end
end


N22_j=zeros(J,N,K);
N22_j1=zeros(J,N,K);
for xunhuan1=1:J
for xunhuan2=1:N
    for xunhuan3=1:K
%   syms x
% N22_j(xunhuan1,xunhuan2,xunhuan3)=-(k_j_k_z(xunhuan1,xunhuan3)./rho_j(xunhuan1))./abs(a)*int( exp(-1j*an(xunhuan2)*x)*cos(k_j_k_x(xunhuan1,xunhuan3)*(x-xj(xunhuan1))), xj(xunhuan1), xj(xunhuan1)+w );
%    f=@(x)exp(-1j*an(xunhuan2)*x).*cos(k_j_k_x(xunhuan1,xunhuan3)*(x-xj(xunhuan1)));
%    N22_j(xunhuan1,xunhuan2,xunhuan3)=-(k_j_k_z(xunhuan1,xunhuan3)./rho_j(xunhuan1))./abs(a)*integral(f, xj(xunhuan1), xj(xunhuan1)+w);
 N22_j(xunhuan1,xunhuan2,xunhuan3)=-(k_j_k_z(xunhuan1,xunhuan3))./abs(a)./rhoc(xunhuan1)*fenbujifen(-1j*an(xunhuan2),k_j_k_x(xunhuan1,xunhuan3),xj(xunhuan1), xj(xunhuan1), xj(xunhuan1)+w(xunhuan1));
    N22_j1(xunhuan1,xunhuan2,xunhuan3)=-(k_j_k_z(xunhuan1,xunhuan3))./abs(a)./rhoc(xunhuan1)*fenbujifen(-1j*an1(xunhuan2),k_j_k_x(xunhuan1,xunhuan3),xj(xunhuan1), xj(xunhuan1), xj(xunhuan1)+w(xunhuan1));

    end
end
end
N22_G=[];
for xunhuan1=1:J
N22_G=[N22_G squeeze(N22_j(xunhuan1,:,:))];
end

N22_G1=[];
for xunhuan1=1:J
N22_G1=[N22_G1 squeeze(N22_j1(xunhuan1,:,:))];
end

M12_j=zeros(J,K,N);
M12_j1=zeros(J,K,N);
for xunhuan1=1:J
for xunhuan2=1:K
    for xunhuan3=1:N
%   syms x
% M12_j(xunhuan1,xunhuan2,xunhuan3)=1/w*int( exp(1j*an(xunhuan3)*x)*cos(k_j_k_x(xunhuan1,xunhuan2)*(x-xj(xunhuan1))), xj(xunhuan1), xj(xunhuan1)+w);
%  f=@(x)exp(1j*an(xunhuan3)*x).*cos(k_j_k_x(xunhuan1,xunhuan2)*(x-xj(xunhuan1)));
%  M12_j(xunhuan1,xunhuan2,xunhuan3)=1/w*integral(f, xj(xunhuan1), xj(xunhuan1)+w);
M12_j(xunhuan1,xunhuan2,xunhuan3)=1/w(xunhuan1)*fenbujifen(1j*an(xunhuan3),k_j_k_x(xunhuan1,xunhuan2),xj(xunhuan1),xj(xunhuan1), xj(xunhuan1)+w(xunhuan1));
M12_j1(xunhuan1,xunhuan2,xunhuan3)=1/w(xunhuan1)*fenbujifen(1j*an1(xunhuan3),k_j_k_x(xunhuan1,xunhuan2),xj(xunhuan1),xj(xunhuan1), xj(xunhuan1)+w(xunhuan1));

    end
end
end
M12_G=[];
for xunhuan1=1:J
M12_G=[M12_G;squeeze(M12_j(xunhuan1,:,:))];
end
M12_G1=[];
for xunhuan1=1:J
M12_G1=[M12_G1;squeeze(M12_j1(xunhuan1,:,:))];
end



M22_j=zeros(J,K,K);
for xunhuan1=1:J
for xunhuan2=1:K
    for xunhuan3=1:K
%   syms x
% M22_j(xunhuan1,xunhuan2,xunhuan3)=1/w*int( cos(k_j_k_x(xunhuan1,xunhuan2)*(x-xj(xunhuan1)))*cos(k_j_k_x(xunhuan1,xunhuan3)*(x-xj(xunhuan1))), xj(xunhuan1), xj(xunhuan1)+w );
%  f=@(x)cos(k_j_k_x(xunhuan1,xunhuan2)*(x-xj(xunhuan1))).*cos(k_j_k_x(xunhuan1,xunhuan3)*(x-xj(xunhuan1)));
 M22_j(xunhuan1,xunhuan2,xunhuan3)=1/w(xunhuan1)*M22jifen(k_j_k_x(xunhuan1,xunhuan2),k_j_k_x(xunhuan1,xunhuan3),xj(xunhuan1), xj(xunhuan1)+w(xunhuan1));
%    M22_j(xunhuan1,xunhuan2,xunhuan3)=1/w*integral(f,xj(xunhuan1), xj(xunhuan1)+w);
 
    end
end
end
% M22_G=blkdiag(squeeze(M22_j(1,:,:)),squeeze(M22_j(2,:,:)),squeeze(M22_j(3,:,:)),squeeze(M22_j(4,:,:)),squeeze(M22_j(5,:,:)),squeeze(M22_j(6,:,:)));

 M22_G=blkdiag(squeeze(M22_j(1,:,:)),squeeze(M22_j(2,:,:)),squeeze(M22_j(3,:,:)));







%解方程

A=[-M12_G M22_G*(I+U_j_k);
    N1 N22_G*(I-U_j_k)];
 B=[M12_G*delta_n;N1*delta_n];


x_=A\B;

A1=[-M12_G1 M22_G*(I+U_j_k);
    N11 N22_G1*(I-U_j_k)];
 B1=[M12_G1*delta_n;N11*delta_n];


x_1=A1\B1;
%方程解分别是
%rn                         tn                       a_j_k         b_j_K
%反射振幅NX1  透射振幅NX1    凹槽里的振幅向下 J*KX1和向上 J*KX1

rn=x_(1:N,:);
rn1=x_1(1:N,:);
rzheng_1(xunhuan_para1,xunhuan_para2)=(rn((N-1)/2));
rzheng0(xunhuan_para1,xunhuan_para2)=(rn((N+1)/2));
rfu0(xunhuan_para1,xunhuan_para2)=(rn1((N+1)/2));
rfu1(xunhuan_para1,xunhuan_para2)=(rn1((N+3)/2));
S=[rzheng0(xunhuan_para1,xunhuan_para2) rfu1(xunhuan_para1,xunhuan_para2);rzheng_1(xunhuan_para1,xunhuan_para2) rfu0(xunhuan_para1,xunhuan_para2)];
eigen(xunhuan_para1,xunhuan_para2,:)=eig((S));
[Eigenvector,Dduijiao]=eig((S));
eigenvector(xunhuan_para1,xunhuan_para2,:,:)=Eigenvector;
% eigenvalue1_(xunhuan_para1,xunhuan_para2,:)=max(eigen(xunhuan_para1,xunhuan_para2,:));
% eigenvalue2_(xunhuan_para1,xunhuan_para2,:)=min(eigen(xunhuan_para1,xunhuan_para2,:));
suan_eigen1_(xunhuan_para1,xunhuan_para2,:)=rzheng0(xunhuan_para1,xunhuan_para2,:)+sqrt(rfu1(xunhuan_para1,xunhuan_para2,:)*rzheng_1(xunhuan_para1,xunhuan_para2,:));
suan_eigen2_(xunhuan_para1,xunhuan_para2,:)=rzheng0(xunhuan_para1,xunhuan_para2,:)-sqrt(rfu1(xunhuan_para1,xunhuan_para2,:)*rzheng_1(xunhuan_para1,xunhuan_para2,:));
B = sort(squeeze(eigen(xunhuan_para1,xunhuan_para2,:)),'ComparisonMethod','real');
eigenvalue1_(xunhuan_para1,xunhuan_para2,:)=B(1,1);
eigenvalue2_(xunhuan_para1,xunhuan_para2,:)=B(2,1);
% xiao_real(xunhuan_para1,xunhuan_para2,:)=min(real(eigen(xunhuan_para1,xunhuan_para2,:)));
% da_real(xunhuan_para1,xunhuan_para2,:)=max(real(eigen(xunhuan_para1,xunhuan_para2,:)));
% xiao_imag(xunhuan_para1,xunhuan_para2,:)=min(imag(eigen(xunhuan_para1,xunhuan_para2,:)));
% da_imag(xunhuan_para1,xunhuan_para2,:)=max(imag(eigen(xunhuan_para1,xunhuan_para2,:)));

% ddd(xunhuan_para1,xunhuan_para2,:)=abs(rzheng0(xunhuan_para1,xunhuan_para2))+abs(abs(rfu1(xunhuan_para1,xunhuan_para2))-1)+abs(rzheng_1(xunhuan_para1,xunhuan_para2))+abs(rfu0(xunhuan_para1,xunhuan_para2));
ddd(xunhuan_para1,xunhuan_para2,:)=abs(rzheng0(xunhuan_para1,xunhuan_para2))+abs(rzheng_1(xunhuan_para1,xunhuan_para2))+abs(rfu0(xunhuan_para1,xunhuan_para2));

    end
end

real_eigenvalue1=real(eigenvalue1_);
imag_eigenvalue1=imag(eigenvalue1_);

real_eigenvalue2=real(eigenvalue2_);
imag_eigenvalue2=imag(eigenvalue2_);

% eigenvalue1=rzheng0+sqrt(rzheng_1.*rfu1);
% eigenvalue2=rzheng0-sqrt(rzheng_1.*rfu1);

% real_eigenvalue1=real(eigenvalue1);
% imag_eigenvalue1=imag(eigenvalue1);
% 
% real_eigenvalue2=real(eigenvalue2);
% imag_eigenvalue2=imag(eigenvalue2);


figure();
[x,y]=meshgrid(cii,crr);
surf(x,y,real_eigenvalue1,imag_eigenvalue1);
hold on;
surf(x,y,real_eigenvalue2,imag_eigenvalue2);
shading(gca,'interp');
xlabel('ci');
ylabel('cr');
 zlabel('real(eigenvalue)');
hold off
colormap(summer)
c = colorbar;
c.Label.String = 'imag(eigenvalue)';


figure();
[x,y]=meshgrid(cii,crr);
surf(x,y,real_eigenvalue1,angle(eigenvalue1_));
hold on;
surf(x,y,real_eigenvalue2,angle(eigenvalue2_));
shading(gca,'interp');
xlabel('ci');
ylabel('cr');
 zlabel('real(eigenvalue)');
hold off
colormap(summer)
c = colorbar;
c.Label.String = 'imag(eigenvalue)';

figure();
% subplot(2,3,1);
% pcolor(x,y,abs(rfu0));
% shading interp;
% % clim([0 0.18]);
% 
% subplot(2,3,2);
% pcolor(x,y,abs(rfu1));
% shading interp;
% % clim([0.94 1]);
% 
% subplot(2,3,3);
% pcolor(x,y,(abs(rzheng_1)));
% shading interp;
% % clim([0 0.09]);
% 
% 
% % figure();
% subplot(2,3,4);
% pcolor(x,y,angle(rfu0)./pi);
% shading interp;
% clim([-1 1]);
% 
% subplot(2,3,5);
% pcolor(x,y,angle(rfu1)./pi);
% shading interp;
% clim([-1 1]);

% subplot(2,3,6);
pcolor(x,y,(angle(rzheng_1))./pi);
shading interp;
clim([-1 1]);
colormap(turbo);

figure();
pcolor(x,y,abs(eigenvalue1_-eigenvalue2_));
shading interp;