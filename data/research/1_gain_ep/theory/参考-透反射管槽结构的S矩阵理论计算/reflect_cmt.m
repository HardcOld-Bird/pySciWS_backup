%%
clc;
clear all;
close all;
f0=3430;
c0=343;
k0=2*pi*f0/c0;
lam=2*pi/k0;
rho = 1.21;
J=3;%单元个数 改的时候要改T_ud_G
% j=1:J;%小写L
thetai=45;%正 S才对
thetar=-thetai;
a=abs(lam/(sind(thetar)-sind(thetai)));%确保第一透射分量是传播模式 a周期长度

%EP
h=[0.569 0.195 0.232]*lam;
w=[0.227 0.115 0.153]*a;%波导宽度
d=[0 0.276 0.070 ]*a;%波导硬边界厚度
%DP
% h=[0.211 0.491 0.21]*lam;
% w=[0.137 0.187 0.145]*a;%波导宽度
% d=[0 0.096 0.081 ]*a;%波导硬边界厚度


%EP
%  cc=[343 343*(1.016+0.0624*1i) 343];
 cc=[343 343*(1.0081+0.0745*1i) 343];
%cc=[343 343*(1.016-0.06226*1i) 343];
%DP
% cc=[343*(1+0.24*1i) 343 343*(1+0.251*1i)];
 kc=2*pi*f0./cc;
rhoc=rho*c0./c0*ones(1,J);
% rhoc=rho*c0./cc;
 n=-10:10;%超表面阶数
N=length(n);
G=2*pi/a;
an=k0*sind(thetai)+n*G;
an1=k0*sind(-thetai)+n*G;
bn=sqrt(k0^2-an.^2);
bn1=sqrt(k0^2-an1.^2);
k=0:1:30;%波导阶数
K=length(k);
k_j_k_x=((1./w)'*(k)*pi);
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

     N1(xunhuan1,xunhuan2)=bn(xunhuan1)./abs(a)./rho*N1jifen(an(xunhuan1),an(xunhuan2),0,a);
     N11(xunhuan1,xunhuan2)=bn1(xunhuan1)./abs(a)./rho*N1jifen(an1(xunhuan1),an1(xunhuan2),0,a);
    end
end


N22_j=zeros(J,N,K);
N22_j1=zeros(J,N,K);
for xunhuan1=1:J
for xunhuan2=1:N
    for xunhuan3=1:K
 N22_j(xunhuan1,xunhuan2,xunhuan3)=(k_j_k_z(xunhuan1,xunhuan3))./abs(a)./rhoc(xunhuan1)*fenbujifen(-1j*an(xunhuan2),k_j_k_x(xunhuan1,xunhuan3),xj(xunhuan1), xj(xunhuan1), xj(xunhuan1)+w(xunhuan1));
    N22_j1(xunhuan1,xunhuan2,xunhuan3)=(k_j_k_z(xunhuan1,xunhuan3))./abs(a)./rhoc(xunhuan1)*fenbujifen(-1j*an1(xunhuan2),k_j_k_x(xunhuan1,xunhuan3),xj(xunhuan1), xj(xunhuan1), xj(xunhuan1)+w(xunhuan1));

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
% tn=x_(1+N:2*N,:);
rzheng_1=(rn((N-1)/2));
rzheng0=(rn((N+1)/2));
rfu0=(rn1((N+1)/2));
rfu1=(rn1((N+3)/2));
S=[rzheng0 rfu1;rzheng_1 rfu0];
abs(S)
eigen=eig((S));
[Eigenvector,Dduijiao]=eig((S));
eigenvector(:,:)=Eigenvector;
eigenvalue1_=max(eigen);
eigenvalue2_=min(eigen);
suan_eigen1_=rzheng0+sqrt(rfu1*rzheng_1);
suan_eigen2_=rzheng0-sqrt(rfu1*rzheng_1);

%画声场
dx = 0.001; 
dz = 0.001; 
x = -2*abs(a):dx:2*abs(a);
% zd = -4*a:dz:-h;
% zh=-h:dz:0;
% zu=0:dz:4*a;
 zu=-2*abs(a):dz:0;
%zu=0:dz:2*abs(a);
zh=0:dz:h;
zd=h:dz:h+2*abs(a);
nx = (max(x)-min(x))/dx+1; 
nx=round(nx);
nuz = (max(zu)-min(zu))/dz+1; 
ndz = (max(zd)-min(zd))/dz+1; 
nzh= (max(zh)-min(zh))/dz+1; 
nzh=round(nzh);
ndz=round(ndz);
nuz=round(nuz);
pu=zeros(N,nuz,nx);%上半部分各阶声场分布  N*x*z
pur=zeros(N,nuz,nx);%上半部分各阶声场分布  N*x*z
puf=zeros(N,nuz,nx);%上半部分各阶声场分布  N*x*z

% pdl1=zeros(L,K,ndz,nx);%下凹槽各阶声场分布   K*x*z

%pu入射+反射  pur入射   puf反射
for xunhuan1=1:N
pu(xunhuan1,:,:)=(delta_n(xunhuan1)*exp(1j*bn(xunhuan1)*zu)+rn(xunhuan1)*exp(-1j*bn(xunhuan1)*zu))'*exp(-1j*an(xunhuan1)*x);
pur(xunhuan1,:,:)=(delta_n(xunhuan1)*exp(1j*bn(xunhuan1)*zu))'*exp(-1j*an(xunhuan1)*x);
puf(xunhuan1,:,:)=(rn(xunhuan1)*exp(-1j*bn(xunhuan1)*zu))'*exp(-1j*an(xunhuan1)*x);
end
pusum=squeeze(sum(pu));
pursum=squeeze(sum(pur));
pufsum=squeeze(sum(puf));


for xunhuan1=1:N
pu1(xunhuan1,:,:)=(delta_n(xunhuan1)*exp(1j*bn1(xunhuan1)*zu)+rn1(xunhuan1)*exp(-1j*bn1(xunhuan1)*zu))'*exp(-1j*an1(xunhuan1)*x);
pur1(xunhuan1,:,:)=(delta_n(xunhuan1)*exp(1j*bn1(xunhuan1)*zu))'*exp(-1j*an1(xunhuan1)*x);
puf1(xunhuan1,:,:)=(rn1(xunhuan1)*exp(-1j*bn1(xunhuan1)*zu))'*exp(-1j*an1(xunhuan1)*x);
end
pusum1=squeeze(sum(pu1));
pursum1=squeeze(sum(pur1));
pufsum1=squeeze(sum(puf1));


%%画图
figure( )
%第一行入射波实部
subplot(2,2,1);
pcolor(x,(zu),(real(pursum))); 
 set(gca,'YDir','reverse');
shading interp;
 clim([-1 1]);
colorbar;
title('matlab入射');




subplot(2,2,2);
pcolor(x,(zu),(real(pufsum))); 
 set(gca,'YDir','reverse');
shading interp;
colormap(jet)
   clim([-1 1]);
colorbar;
title('matlab反射');


subplot(2,2,3);
pcolor(x,(zu),(real(pursum1))); 
set(gca,'YDir','reverse');
shading interp;
 clim([-1 1]);
colorbar;
title('matlab入射');



%第二行反射波实部
subplot(2,2,4);
pcolor(x,(zu),(real(pufsum1))); 
 set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 clim([-1 1]);
colorbar;
title('matlab反射');

rzheng_1=(rn((N-1)/2));
rzheng0=(rn((N+1)/2));
rfu0=(rn1((N+1)/2));
rfu1=(rn1((N+3)/2));

aaar=abs(rn((N+1)/2)).^2+abs(rn((N-1)/2)).^2;
aaar1=abs(rn1((N+1)/2)).^2+abs(rn1((N+3)/2)).^2;
eigenvalue1=rzheng0+sqrt(rzheng_1.*rfu1);
eigenvalue2=rzheng0-sqrt(rzheng_1.*rfu1);

real_eigenvalue1=real(eigenvalue1);