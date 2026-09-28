%%
%透射反射 4阶矩阵 单个可以试 
%考虑波导n阶CMT
%0.893180017143357	0.210446030454783	0.0623196723089962	0.0532366221550005	0.131263814832633	0.331852240931967	0.210349230254252	0.449082244904531

% clc;
% clear all;
% close all;
f0=3430; 
c0=343;
k0=2*pi*f0/c0;
lam=2*pi/k0;
thetai=45;
thetat=-45;
a=lam/abs(sind(thetat)-sind(thetai));%确保第一透射分量是传播模式
L=2;%改的时候要改T_ud_G
n=-10:10;
N=length(n);
G=2*pi/a;
an=k0*sind(thetai)+n*G;
an1=k0*sind(-thetai)+n*G;
bn=sqrt(k0^2-an.^2);
bn1=sqrt(k0^2-an1.^2);

k=0:15;
K=length(k);
x1=[0.893180017143357 0.210446030454783	0.0623196723089962 0.0532366221550005 0.131263814832633	0.331852240931967 0.210349230254252	0.449082244904531];
hc=x1(1)*lam;%通管长度
tc=x1(2)*a;%通管宽度 
wlu=x1(3:4)*a;%上间距
 d_u_l=x1(5:6)*lam;
tlu=x1(7:8)*a;%上管槽宽度
wld=wlu;
% % dddx=-0.024;
% % dddy=-0.005;
% %  d_u_l(1)=((1-dddx)*x1(5))*lam;
% %  d_d_l(1)=((1-dddx)*x1(5))*lam;
% % d_u_l(2)=((1-dddy)*x1(6))*lam;
% % d_d_l(2)=((1-dddy)*x1(6))*lam;
% % dddz=-0.025;
% % tc=(x1(2)*(1-dddz))*a;
% % wlu(1)=(x1(3)+dddz*x1(2))*a;
% % wld(1)=(x1(3)+dddz*x1(2))*a;
d_d_l=d_u_l;
tld=tlu;
panduan1=a-sum(wlu)-sum(tlu)-tc;


akc=((1./tc)'*(k)*pi);
thegemkc=sqrt(k0^2-akc.^2);
aklu=((1./tlu)'*(k)*pi);
akld=((1./tld)'*(k)*pi);

thegemklu=sqrt((k0'*ones(1,K)).^2-aklu.^2);
thegemkld=sqrt((k0'*ones(1,K)).^2-akld.^2);




xc=a-tc;

xlu=ones(1,L);
xlu(1)=wlu(1);
for xunhuan1=2:L
xlu(xunhuan1)=xlu(xunhuan1-1)+tlu(xunhuan1-1)+wlu(xunhuan1);
end
xld=ones(1,L);
xld(1)=wld(1);
for xunhuan1=2:L
xld(xunhuan1)=xld(xunhuan1-1)+tld(xunhuan1-1)+wld(xunhuan1);
end

Auzheng=[zeros(1,(N-1)/2) 1 zeros(1,(N-1)/2)]';
 Adfu=zeros(1,length(Auzheng))';
 T_u_l=zeros(L,K,K);
 for xunhuan1=1:L
for xunhuan2=1:K
T_u_l(xunhuan1,xunhuan2,xunhuan2)=exp(1j*2*thegemklu(xunhuan1,xunhuan2).*d_u_l(xunhuan1));
end
 end
T_u_G=  blkdiag(squeeze(T_u_l(1,:,:)),squeeze(T_u_l(2,:,:)));

T_d_l=zeros(L,K,K);
 for xunhuan1=1:L
for xunhuan2=1:K
T_d_l(xunhuan1,xunhuan2,xunhuan2)=exp(1j*2*thegemkld(xunhuan1,xunhuan2).*d_d_l(xunhuan1));
end
 end
T_d_G=  blkdiag(squeeze(T_d_l(1,:,:)),squeeze(T_d_l(2,:,:)));






I=eye(K*L,K*L);%大写i

N1=ones(N,N);
N11=ones(N,N);
for xunhuan1=1:N
    for xunhuan2=1:N
 N1(xunhuan1,xunhuan2)=-bn(xunhuan1)./abs(a)*N1jifen(an(xunhuan1),an(xunhuan2),0,a);
  N11(xunhuan1,xunhuan2)=-bn1(xunhuan1)./abs(a)*N1jifen(an1(xunhuan1),an1(xunhuan2),0,a);
 
    end
end
N21_c=zeros(N,K);
N21_c1=zeros(N,K);
for xunhuan1=1:N
    for xunhuan2=1:K
    N21_c(xunhuan1,xunhuan2)=(-thegemkc(xunhuan2)/abs(a))*fenbujifen(-1j*an(xunhuan1),akc(xunhuan2),xc, xc, xc+tc);
        N21_c1(xunhuan1,xunhuan2)=(-thegemkc(xunhuan2)/abs(a))*fenbujifen(-1j*an1(xunhuan1),akc(xunhuan2),xc, xc, xc+tc);

    end
end

N22_lu=zeros(L,N,K);
N22_ld=zeros(L,N,K);
N22_lu1=zeros(L,N,K);
N22_ld1=zeros(L,N,K);
for xunhuan1=1:L
for xunhuan2=1:N
    for xunhuan3=1:K
N22_lu(xunhuan1,xunhuan2,xunhuan3)=(-thegemklu(xunhuan1,xunhuan3)/abs(a))*fenbujifen(-1j*an(xunhuan2),aklu(xunhuan1,xunhuan3),xlu(xunhuan1), xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1) );
N22_ld(xunhuan1,xunhuan2,xunhuan3)=(-thegemkld(xunhuan1,xunhuan3)/abs(a))*fenbujifen(-1j*an(xunhuan2),akld(xunhuan1,xunhuan3),xld(xunhuan1), xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1) );
N22_lu1(xunhuan1,xunhuan2,xunhuan3)=(-thegemklu(xunhuan1,xunhuan3)/abs(a))*fenbujifen(-1j*an1(xunhuan2),aklu(xunhuan1,xunhuan3),xlu(xunhuan1), xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1) );
N22_ld1(xunhuan1,xunhuan2,xunhuan3)=(-thegemkld(xunhuan1,xunhuan3)/abs(a))*fenbujifen(-1j*an1(xunhuan2),akld(xunhuan1,xunhuan3),xld(xunhuan1), xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1) );

    end
end
end
N22_Gu=[];
for xunhuan1=1:L
N22_Gu=[N22_Gu squeeze(N22_lu(xunhuan1,:,:))];
end
N22_Gd=[];
for xunhuan1=1:L
N22_Gd=[N22_Gd squeeze(N22_ld(xunhuan1,:,:))];
end

N22_Gu1=[];
for xunhuan1=1:L
N22_Gu1=[N22_Gu1 squeeze(N22_lu1(xunhuan1,:,:))];
end
N22_Gd1=[];
for xunhuan1=1:L
N22_Gd1=[N22_Gd1 squeeze(N22_ld1(xunhuan1,:,:))];
end


M12_lu=zeros(L,K,N);
M12_ld=zeros(L,K,N);
M12_lu1=zeros(L,K,N);
M12_ld1=zeros(L,K,N);
for xunhuan1=1:L
for xunhuan2=1:K
    for xunhuan3=1:N
M12_lu(xunhuan1,xunhuan2,xunhuan3)=1/tlu(xunhuan1)*fenbujifen(1j*an(xunhuan3),aklu(xunhuan1,xunhuan2),xlu(xunhuan1),xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1));
M12_ld(xunhuan1,xunhuan2,xunhuan3)=1/tld(xunhuan1)*fenbujifen(1j*an(xunhuan3),akld(xunhuan1,xunhuan2),xld(xunhuan1),xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1));
M12_lu1(xunhuan1,xunhuan2,xunhuan3)=1/tlu(xunhuan1)*fenbujifen(1j*an1(xunhuan3),aklu(xunhuan1,xunhuan2),xlu(xunhuan1),xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1));
M12_ld1(xunhuan1,xunhuan2,xunhuan3)=1/tld(xunhuan1)*fenbujifen(1j*an1(xunhuan3),akld(xunhuan1,xunhuan2),xld(xunhuan1),xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1));

    end
end
end
M12_Gu=[];
for xunhuan1=1:L
M12_Gu=[M12_Gu;squeeze(M12_lu(xunhuan1,:,:))];
end
M12_Gd=[];
for xunhuan1=1:L
M12_Gd=[M12_Gd;squeeze(M12_ld(xunhuan1,:,:))];
end
M12_Gu1=[];
for xunhuan1=1:L
M12_Gu1=[M12_Gu1;squeeze(M12_lu1(xunhuan1,:,:))];
end
M12_Gd1=[];
for xunhuan1=1:L
M12_Gd1=[M12_Gd1;squeeze(M12_ld1(xunhuan1,:,:))];
end


M22_lu=zeros(L,K,K);
M22_ld=zeros(L,K,K);
for xunhuan1=1:L
for xunhuan2=1:K
    for xunhuan3=1:K
  M22_lu(xunhuan1,xunhuan2,xunhuan3)=1/tlu(xunhuan1)*M22jifen(aklu(xunhuan1,xunhuan2),aklu(xunhuan1,xunhuan3), xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1));
    M22_ld(xunhuan1,xunhuan2,xunhuan3)=1/tld(xunhuan1)*M22jifen(akld(xunhuan1,xunhuan2),akld(xunhuan1,xunhuan3), xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1));

    end
end
end
 M22_Gu=blkdiag(squeeze(M22_lu(1,:,:)),squeeze(M22_lu(2,:,:)));
 M22_Gd=blkdiag(squeeze(M22_ld(1,:,:)),squeeze(M22_ld(2,:,:)));

 
M11_c=zeros(K,N);
M11_c1=zeros(K,N);
for xunhuan1=1:K
for xunhuan2=1:N
M11_c(xunhuan1,xunhuan2)=1/tc*fenbujifen(1j*an(xunhuan2),akc(xunhuan1),xc, xc, xc+tc);
M11_c1(xunhuan1,xunhuan2)=1/tc*fenbujifen(1j*an1(xunhuan2),akc(xunhuan1),xc, xc, xc+tc);

end
end

M21_c=zeros(K,K);
for xunhuan1=1:K
for xunhuan2=1:K
M21_c(xunhuan1,xunhuan2)=1/tc*M22jifen(akc(xunhuan1),akc(xunhuan2), xc, xc+tc);
end
end

Tc=zeros(K,K);
for xunhuan1=1:K
Tc(xunhuan1,xunhuan1)=exp(1j*thegemkc(xunhuan1)*hc);
end


%解方程

A=[-M11_c zeros(K,N) M21_c M21_c*Tc zeros(K,L*K) zeros(K,L*K);
    zeros(K,N) -M11_c M21_c*Tc M21_c zeros(K,L*K) zeros(K,L*K);
    -M12_Gu zeros(L*K,N) zeros(L*K,K) zeros(L*K,K) M22_Gu*(I+T_u_G)  zeros(L*K,L*K);
     N1 zeros(N,N) N21_c -N21_c*Tc N22_Gu*(I-T_u_G) zeros(N,L*K);
zeros(L*K,N) -M12_Gd zeros(L*K,K) zeros(K*L,K) zeros(K*L,L*K) M22_Gd*(T_d_G+I);
 zeros(N,N) N1 -N21_c*Tc N21_c zeros(N,L*K) -N22_Gd*(T_d_G-I)];
 B=[M11_c*Auzheng;M11_c*Adfu;M12_Gu*Auzheng;N1*Auzheng;M12_Gd*Adfu;N1*Adfu];

 A1=[-M11_c1 zeros(K,N) M21_c M21_c*Tc zeros(K,L*K) zeros(K,L*K);
    zeros(K,N) -M11_c1 M21_c*Tc M21_c zeros(K,L*K) zeros(K,L*K);
    -M12_Gu1 zeros(L*K,N) zeros(L*K,K) zeros(L*K,K) M22_Gu*(I+T_u_G)  zeros(L*K,L*K);
     N11 zeros(N,N) N21_c1 -N21_c1*Tc N22_Gu1*(I-T_u_G) zeros(N,L*K);
zeros(L*K,N) -M12_Gd1 zeros(L*K,K) zeros(K*L,K) zeros(K*L,L*K) M22_Gd*(T_d_G+I);
 zeros(N,N) N11 -N21_c1*Tc N21_c1 zeros(N,L*K) -N22_Gd1*(T_d_G-I)];
 B1=[M11_c1*Auzheng;M11_c1*Adfu;M12_Gu1*Auzheng;N11*Auzheng;M12_Gd1*Adfu;N11*Adfu];


x_=A\B;

x_1=A1\B1;




%方程解分别是
%Aufu                Adzheng               Hczheng HcfuP                               HuGzheng HdGfu
%反射振幅NX1  透射振幅NX1    channel里振幅 向下KX1和向上KX1    凹槽里的振幅u L*KX1和d L*KX1

Aufu=x_(1:N,:);
Adzheng=x_(1+N:2*N,:);
Aufu1=x_1(1:N,:);
Adzheng1=x_1(1+N:2*N,:);

u_L_r0=Aufu((N+1)/2);
u_L_rfu1=Aufu((N-1)/2);
u_R_r0=Aufu1((N+1)/2);
u_R_rzheng1=Aufu1((N+3)/2);
u_L_t0=Adzheng((N+1)/2);
u_L_tfu1=Adzheng((N-1)/2);
u_R_t0=Adzheng1((N+1)/2);
u_R_tzheng1=Adzheng1((N+3)/2);
% aaat=abs(Adzheng((N-1)/2)).^2+abs(Adzheng((N+1)/2)).^2;
% aaar=abs(Aufu((N-1)/2)).^2+abs(Aufu((N+1)/2)).^2;
% aaat+aaar
% aaat1=abs(Adzheng1((N+1)/2)).^2+abs(Adzheng1((N+3)/2)).^2;
% aaar1=abs(Aufu1((N+1)/2)).^2+abs(Aufu1((N+3)/2)).^2;




dx = 0.001; 
dz = 0.001; 
x = -2*a:dx:2*a;

zu=-2*a:dz:0;
zd=hc:dz:hc+2*a;
nx = (max(x)-min(x))/dx+1; 
nx=round(nx);
nuz = (max(zu)-min(zu))/dz+1; 
ndz = (max(zd)-min(zd))/dz+1; 
ndz=round(ndz);
nuz=round(nuz);
pur=zeros(N,nuz,nx);%上半部分各阶声场分布  N*x*z
puf=zeros(N,nuz,nx);%上半部分各阶声场分布  N*x*z
pd=zeros(N,ndz,nx);%下半部分各阶声场分布  N*x*z

% pdl1=zeros(L,K,ndz,nx);%下凹槽各阶声场分布   K*x*z

%pu入射+反射  pur入射   puf反射
for xunhuan1=1:N
pur(xunhuan1,:,:)=(Auzheng(xunhuan1)*exp(1j*bn(xunhuan1)*zu))'*exp(-1j*an(xunhuan1)*x);
puf(xunhuan1,:,:)=(Aufu(xunhuan1)*exp(-1j*bn(xunhuan1)*zu))'*exp(-1j*an(xunhuan1)*x);
end
pursum=squeeze(sum(pur));
pufsum=squeeze(sum(puf));
for xunhuan1=1:N
    pd(xunhuan1,:,:)=(Adzheng(xunhuan1)*exp(1j*bn(xunhuan1)*(zd)))'*exp(-1j*an(xunhuan1)*x);    
end
pdsum=squeeze(sum(pd));



for xunhuan1=1:N
pur1(xunhuan1,:,:)=(Auzheng(xunhuan1)*exp(1j*bn1(xunhuan1)*zu))'*exp(-1j*an1(xunhuan1)*x);
puf1(xunhuan1,:,:)=(Aufu1(xunhuan1)*exp(-1j*bn1(xunhuan1)*zu))'*exp(-1j*an1(xunhuan1)*x);
end
pursum1=squeeze(sum(pur1));
pufsum1=squeeze(sum(puf1));
for xunhuan1=1:N
    pd1(xunhuan1,:,:)=(Adzheng1(xunhuan1)*exp(1j*bn1(xunhuan1)*(zd)))'*exp(-1j*an1(xunhuan1)*x);    
end
pdsum1=squeeze(sum(pd1));










%%画图
figure( )
%第一行入射波实部虚部
subplot(3,2,1);
pcolor(x,zu,real(pursum)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('左上入射实部');


subplot(3,2,3);
pcolor(x,zu,real(pufsum)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('左上反射实部');


subplot(3,2,5);
pcolor(x,zd,real(pdsum)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('左上透射实部');



%第一行入射波实部虚部
subplot(3,2,2);
pcolor(x,zu,real(pursum1)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('右上入射实部');


subplot(3,2,4);
pcolor(x,zu,real(pufsum1)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('右上反射实部');


subplot(3,2,6);
pcolor(x,zd,real(pdsum1)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('右上透射实部');





%下入射调换
zhongjian=wlu;
wlu=wld;
wld=zhongjian;
zhongjian=tlu;
tlu=tld;
tld=zhongjian;
zhongjian=d_u_l;
d_u_l=d_d_l;
d_d_l=zhongjian;
% wld=0.05*a*ones(1,L+1);%上间距
% wlu=0.05*a*ones(1,L+1);%下间距
% tld=(a-tc-(L+1)*0.05*a)/L*ones(1,L);%上管槽宽度
% tlu=(a-tc-(L+1)*0.05*a)/L*ones(1,L);%下管槽宽度
akc=((1./tc)'*(k)*pi);
thegemkc=sqrt(k0^2-akc.^2);
aklu=((1./tlu)'*(k)*pi);
akld=((1./tld)'*(k)*pi);

 thegemklu=sqrt((k0'*ones(1,K)).^2-aklu.^2);
  thegemkld=sqrt((k0'*ones(1,K)).^2-akld.^2);


% d_d_l=[0.172913 0.203262 0.179856 0.218711 0.042842]*lam;
% d_u_l=[0.122602 0.017135 0.000647 0.000052 0.288311]*lam;


xc=a-tc;

xlu=ones(1,L);
xlu(1)=wlu(1);
for xunhuan1=2:L
xlu(xunhuan1)=xlu(xunhuan1-1)+tlu(xunhuan1-1)+wlu(xunhuan1);
end
xld=ones(1,L);
xld(1)=wld(1);
for xunhuan1=2:L
xld(xunhuan1)=xld(xunhuan1-1)+tld(xunhuan1-1)+wld(xunhuan1);
end

Auzheng=[zeros(1,(N-1)/2) 1 zeros(1,(N-1)/2)]';
 Adfu=zeros(1,length(Auzheng))';
 T_u_l=zeros(L,K,K);
 for xunhuan1=1:L
for xunhuan2=1:K
T_u_l(xunhuan1,xunhuan2,xunhuan2)=exp(1j*2*thegemklu(xunhuan1,xunhuan2).*d_u_l(xunhuan1));
end
 end
T_u_G=  blkdiag(squeeze(T_u_l(1,:,:)),squeeze(T_u_l(2,:,:)));

T_d_l=zeros(L,K,K);
 for xunhuan1=1:L
for xunhuan2=1:K
T_d_l(xunhuan1,xunhuan2,xunhuan2)=exp(1j*2*thegemkld(xunhuan1,xunhuan2).*d_d_l(xunhuan1));
end
 end
T_d_G=  blkdiag(squeeze(T_d_l(1,:,:)),squeeze(T_d_l(2,:,:)));






I=eye(K*L,K*L);%大写i

N1=ones(N,N);
N11=ones(N,N);
for xunhuan1=1:N
    for xunhuan2=1:N
 N1(xunhuan1,xunhuan2)=-bn(xunhuan1)./abs(a)*N1jifen(an(xunhuan1),an(xunhuan2),0,a);
  N11(xunhuan1,xunhuan2)=-bn1(xunhuan1)./abs(a)*N1jifen(an1(xunhuan1),an1(xunhuan2),0,a);
 
    end
end
N21_c=zeros(N,K);
N21_c1=zeros(N,K);
for xunhuan1=1:N
    for xunhuan2=1:K
    N21_c(xunhuan1,xunhuan2)=(-thegemkc(xunhuan2)/abs(a))*fenbujifen(-1j*an(xunhuan1),akc(xunhuan2),xc, xc, xc+tc);
        N21_c1(xunhuan1,xunhuan2)=(-thegemkc(xunhuan2)/abs(a))*fenbujifen(-1j*an1(xunhuan1),akc(xunhuan2),xc, xc, xc+tc);

    end
end

N22_lu=zeros(L,N,K);
N22_ld=zeros(L,N,K);
N22_lu1=zeros(L,N,K);
N22_ld1=zeros(L,N,K);
for xunhuan1=1:L
for xunhuan2=1:N
    for xunhuan3=1:K
N22_lu(xunhuan1,xunhuan2,xunhuan3)=(-thegemklu(xunhuan1,xunhuan3)/abs(a))*fenbujifen(-1j*an(xunhuan2),aklu(xunhuan1,xunhuan3),xlu(xunhuan1), xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1) );
N22_ld(xunhuan1,xunhuan2,xunhuan3)=(-thegemkld(xunhuan1,xunhuan3)/abs(a))*fenbujifen(-1j*an(xunhuan2),akld(xunhuan1,xunhuan3),xld(xunhuan1), xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1) );
N22_lu1(xunhuan1,xunhuan2,xunhuan3)=(-thegemklu(xunhuan1,xunhuan3)/abs(a))*fenbujifen(-1j*an1(xunhuan2),aklu(xunhuan1,xunhuan3),xlu(xunhuan1), xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1) );
N22_ld1(xunhuan1,xunhuan2,xunhuan3)=(-thegemkld(xunhuan1,xunhuan3)/abs(a))*fenbujifen(-1j*an1(xunhuan2),akld(xunhuan1,xunhuan3),xld(xunhuan1), xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1) );

    end
end
end
N22_Gu=[];
for xunhuan1=1:L
N22_Gu=[N22_Gu squeeze(N22_lu(xunhuan1,:,:))];
end
N22_Gd=[];
for xunhuan1=1:L
N22_Gd=[N22_Gd squeeze(N22_ld(xunhuan1,:,:))];
end

N22_Gu1=[];
for xunhuan1=1:L
N22_Gu1=[N22_Gu1 squeeze(N22_lu1(xunhuan1,:,:))];
end
N22_Gd1=[];
for xunhuan1=1:L
N22_Gd1=[N22_Gd1 squeeze(N22_ld1(xunhuan1,:,:))];
end


M12_lu=zeros(L,K,N);
M12_ld=zeros(L,K,N);
M12_lu1=zeros(L,K,N);
M12_ld1=zeros(L,K,N);
for xunhuan1=1:L
for xunhuan2=1:K
    for xunhuan3=1:N
M12_lu(xunhuan1,xunhuan2,xunhuan3)=1/tlu(xunhuan1)*fenbujifen(1j*an(xunhuan3),aklu(xunhuan1,xunhuan2),xlu(xunhuan1),xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1));
M12_ld(xunhuan1,xunhuan2,xunhuan3)=1/tld(xunhuan1)*fenbujifen(1j*an(xunhuan3),akld(xunhuan1,xunhuan2),xld(xunhuan1),xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1));
M12_lu1(xunhuan1,xunhuan2,xunhuan3)=1/tlu(xunhuan1)*fenbujifen(1j*an1(xunhuan3),aklu(xunhuan1,xunhuan2),xlu(xunhuan1),xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1));
M12_ld1(xunhuan1,xunhuan2,xunhuan3)=1/tld(xunhuan1)*fenbujifen(1j*an1(xunhuan3),akld(xunhuan1,xunhuan2),xld(xunhuan1),xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1));

    end
end
end
M12_Gu=[];
for xunhuan1=1:L
M12_Gu=[M12_Gu;squeeze(M12_lu(xunhuan1,:,:))];
end
M12_Gd=[];
for xunhuan1=1:L
M12_Gd=[M12_Gd;squeeze(M12_ld(xunhuan1,:,:))];
end
M12_Gu1=[];
for xunhuan1=1:L
M12_Gu1=[M12_Gu1;squeeze(M12_lu1(xunhuan1,:,:))];
end
M12_Gd1=[];
for xunhuan1=1:L
M12_Gd1=[M12_Gd1;squeeze(M12_ld1(xunhuan1,:,:))];
end


M22_lu=zeros(L,K,K);
M22_ld=zeros(L,K,K);
for xunhuan1=1:L
for xunhuan2=1:K
    for xunhuan3=1:K
  M22_lu(xunhuan1,xunhuan2,xunhuan3)=1/tlu(xunhuan1)*M22jifen(aklu(xunhuan1,xunhuan2),aklu(xunhuan1,xunhuan3), xlu(xunhuan1), xlu(xunhuan1)+tlu(xunhuan1));
    M22_ld(xunhuan1,xunhuan2,xunhuan3)=1/tld(xunhuan1)*M22jifen(akld(xunhuan1,xunhuan2),akld(xunhuan1,xunhuan3), xld(xunhuan1), xld(xunhuan1)+tld(xunhuan1));

    end
end
end
 M22_Gu=blkdiag(squeeze(M22_lu(1,:,:)),squeeze(M22_lu(2,:,:)));
 M22_Gd=blkdiag(squeeze(M22_ld(1,:,:)),squeeze(M22_ld(2,:,:)));

 
M11_c=zeros(K,N);
M11_c1=zeros(K,N);
for xunhuan1=1:K
for xunhuan2=1:N
M11_c(xunhuan1,xunhuan2)=1/tc*fenbujifen(1j*an(xunhuan2),akc(xunhuan1),xc, xc, xc+tc);
M11_c1(xunhuan1,xunhuan2)=1/tc*fenbujifen(1j*an1(xunhuan2),akc(xunhuan1),xc, xc, xc+tc);

end
end

M21_c=zeros(K,K);
for xunhuan1=1:K
for xunhuan2=1:K
M21_c(xunhuan1,xunhuan2)=1/tc*M22jifen(akc(xunhuan1),akc(xunhuan2), xc, xc+tc);
end
end

Tc=zeros(K,K);
for xunhuan1=1:K
Tc(xunhuan1,xunhuan1)=exp(1j*thegemkc(xunhuan1)*hc);
end


%解方程

A=[-M11_c zeros(K,N) M21_c M21_c*Tc zeros(K,L*K) zeros(K,L*K);
    zeros(K,N) -M11_c M21_c*Tc M21_c zeros(K,L*K) zeros(K,L*K);
    -M12_Gu zeros(L*K,N) zeros(L*K,K) zeros(L*K,K) M22_Gu*(I+T_u_G)  zeros(L*K,L*K);
     N1 zeros(N,N) N21_c -N21_c*Tc N22_Gu*(I-T_u_G) zeros(N,L*K);
zeros(L*K,N) -M12_Gd zeros(L*K,K) zeros(K*L,K) zeros(K*L,L*K) M22_Gd*(T_d_G+I);
 zeros(N,N) N1 -N21_c*Tc N21_c zeros(N,L*K) -N22_Gd*(T_d_G-I)];
 B=[M11_c*Auzheng;M11_c*Adfu;M12_Gu*Auzheng;N1*Auzheng;M12_Gd*Adfu;N1*Adfu];

 A1=[-M11_c1 zeros(K,N) M21_c M21_c*Tc zeros(K,L*K) zeros(K,L*K);
    zeros(K,N) -M11_c1 M21_c*Tc M21_c zeros(K,L*K) zeros(K,L*K);
    -M12_Gu1 zeros(L*K,N) zeros(L*K,K) zeros(L*K,K) M22_Gu*(I+T_u_G)  zeros(L*K,L*K);
     N11 zeros(N,N) N21_c1 -N21_c1*Tc N22_Gu1*(I-T_u_G) zeros(N,L*K);
zeros(L*K,N) -M12_Gd1 zeros(L*K,K) zeros(K*L,K) zeros(K*L,L*K) M22_Gd*(T_d_G+I);
 zeros(N,N) N11 -N21_c1*Tc N21_c1 zeros(N,L*K) -N22_Gd1*(T_d_G-I)];
 B1=[M11_c1*Auzheng;M11_c1*Adfu;M12_Gu1*Auzheng;N11*Auzheng;M12_Gd1*Adfu;N11*Adfu];


x_=A\B;

x_1=A1\B1;




%方程解分别是
%Aufu                Adzheng               Hczheng HcfuP                               HuGzheng HdGfu
%反射振幅NX1  透射振幅NX1    channel里振幅 向下KX1和向上KX1    凹槽里的振幅u L*KX1和d L*KX1

Aufuzuoxia=x_(1:N,:);%左下入射的透射
Adzhengzuoxia=x_(1+N:2*N,:);%左下入射的反射
Aufu1youxia=x_1(1:N,:);%右下入射的透射
Adzheng1youxia=x_1(1+N:2*N,:);%右下入射的透射


d_L_r0=Aufuzuoxia((N+1)/2);
d_L_rfu1=Aufuzuoxia((N-1)/2);
d_R_r0=Aufu1youxia((N+1)/2);
d_R_rzheng1=Aufu1youxia((N+3)/2);
d_L_t0=Adzhengzuoxia((N+1)/2);
d_L_tfu1=Adzhengzuoxia((N-1)/2);
d_R_t0=Adzheng1youxia((N+1)/2);
d_R_tzheng1=Adzheng1youxia((N+3)/2);

% aaatxia=abs(Adzhengzuoxia((N-1)/2)).^2+abs(Adzhengzuoxia((N+1)/2)).^2;
% aaarxia=abs(Aufuzuoxia((N-1)/2)).^2+abs(Aufuzuoxia((N+1)/2)).^2;
% aaat1xia=abs(Adzheng1youxia((N+1)/2)).^2+abs(Adzheng1youxia((N+3)/2)).^2;
% aaar1xia=abs(Aufu1youxia((N+1)/2)).^2+abs(Aufu1youxia((N+3)/2)).^2;
% s2(4,:)=[u_L_rfu1  u_R_r0       d_L_tfu1  d_R_t0];
% s2(3,:)=[u_L_r0    u_R_rzheng1  d_L_t0    d_R_tzheng1];
% s2(2,:)=[u_L_tfu1  u_R_t0       d_L_rfu1  d_R_r0];
% s2(1,:)=[u_L_t0    u_R_tzheng1  d_L_r0    d_R_rzheng1];

s2(1,:)=[u_L_tfu1  u_R_t0       d_R_r0      d_L_rfu1];
s2(2,:)=[u_L_t0    u_R_tzheng1  d_R_rzheng1 d_L_r0    ];
s2(3,:)=[u_L_r0    u_R_rzheng1  d_R_tzheng1 d_L_t0    ];
s2(4,:)=[u_L_rfu1  u_R_r0       d_R_t0      d_L_tfu1];
eigens2=eig(s2);
[Eigenvector_s2,Dduijiao_s2]=eig(s2);
quanvectoe_panduan(1)=Eigenvector_s2(:,1)'/Eigenvector_s2(:,2)';
quanvectoe_panduan(2)=Eigenvector_s2(:,1)'/Eigenvector_s2(:,3)';
quanvectoe_panduan(3)=Eigenvector_s2(:,1)'/Eigenvector_s2(:,4)';
quanvectoe_panduan(4)=Eigenvector_s2(:,2)'/Eigenvector_s2(:,3)';
quanvectoe_panduan(5)=Eigenvector_s2(:,2)'/Eigenvector_s2(:,4)';
quanvectoe_panduan(6)=Eigenvector_s2(:,3)'/Eigenvector_s2(:,4)';
% quanvectoe_panduan
upperspace=[u_L_r0 u_R_rzheng1;u_L_rfu1 u_R_r0];
abs(upperspace)

eigenupper=eig(upperspace)

[Eigenvector_upper,Dduijiao_upper]=eig(upperspace);
leftspace=[u_L_tfu1 d_L_rfu1;u_L_rfu1 d_L_tfu1];
eigenleft=eig(leftspace);
[Eigenvector_left,Dduijiao_left]=eig(leftspace);
rightspace=[u_R_tzheng1 d_R_rzheng1;u_R_rzheng1 d_R_tzheng1];
eigenright=eig(rightspace);
[Eigenvector_right,Dduijiao_right]=eig(rightspace);

B_upper= sort((eigenupper),'ComparisonMethod','real');
eigenvalue1_upper=B_upper(1,1);
eigenvalue2_upper=B_upper(2,1);
real_eigenvalue1_upper=real(eigenvalue1_upper);
imag_eigenvalue1_upper=imag(eigenvalue1_upper);
real_eigenvalue2_upper=real(eigenvalue2_upper);
imag_eigenvalue2_upper=imag(eigenvalue2_upper);



huatu1=abs(real_eigenvalue1_upper-real_eigenvalue2_upper);

huatu2=abs(imag_eigenvalue1_upper-imag_eigenvalue2_upper);



dx = 0.001; 
dz = 0.001; 
x = -2*a:dx:2*a;

zu=-2*a:dz:0;
zd=hc:dz:hc+2*a;
nx = (max(x)-min(x))/dx+1; 
nx=round(nx);
nuz = (max(zu)-min(zu))/dz+1; 
ndz = (max(zd)-min(zd))/dz+1; 
ndz=round(ndz);
nuz=round(nuz);
pur=zeros(N,nuz,nx);%上半部分各阶声场分布  N*x*z
puf=zeros(N,nuz,nx);%上半部分各阶声场分布  N*x*z
pd=zeros(N,ndz,nx);%下半部分各阶声场分布  N*x*z

% pdl1=zeros(L,K,ndz,nx);%下凹槽各阶声场分布   K*x*z

%pu入射+反射  pur入射   puf反射
for xunhuan1=1:N
pur(xunhuan1,:,:)=(Auzheng(xunhuan1)*exp(-1j*bn(xunhuan1)*zu))'*exp(-1j*an(xunhuan1)*x);
puf(xunhuan1,:,:)=(Aufuzuoxia(xunhuan1)*exp(-1j*bn(xunhuan1)*zu))'*exp(1j*an(xunhuan1)*x);
end
pursum=squeeze(sum(pur));
pufsum=squeeze(sum(puf));
for xunhuan1=1:N
    pd(xunhuan1,:,:)=(Adzhengzuoxia(xunhuan1)*exp(1j*bn(xunhuan1)*(zd)))'*exp(1j*an(xunhuan1)*x);    
end
pdsum=squeeze(sum(pd));



for xunhuan1=1:N
pur1(xunhuan1,:,:)=(Auzheng(xunhuan1)*exp(-1j*bn1(xunhuan1)*zu))'*exp(-1j*an1(xunhuan1)*x);
puf1(xunhuan1,:,:)=(Aufu1youxia(xunhuan1)*exp(-1j*bn1(xunhuan1)*zu))'*exp(1j*an1(xunhuan1)*x);
end
pursum1=squeeze(sum(pur1));
pufsum1=squeeze(sum(puf1));
for xunhuan1=1:N
    pd1(xunhuan1,:,:)=(Adzheng1youxia(xunhuan1)*exp(1j*bn1(xunhuan1)*(zd)))'*exp(1j*an1(xunhuan1)*x);    
end
pdsum1=squeeze(sum(pd1));










%%画图
figure( )
%第一行入射波实部虚部
subplot(3,2,1);
pcolor(x,zu,real(pursum)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('左下入射实部');


subplot(3,2,3);
pcolor(x,zu,real(pdsum)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('左下透射实部');


subplot(3,2,5);
pcolor(x,zd,real(pufsum)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('左下反射实部');



%第一行入射波实部虚部
subplot(3,2,2);
pcolor(x,zu,real(pursum1)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('右下入射实部');


subplot(3,2,4);
pcolor(x,zu,real(pdsum1)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('右下透射实部');


subplot(3,2,6);
pcolor(x,zd,real(pufsum1)); 
set(gca,'YDir','reverse');
shading interp;
colormap(jet)
 caxis([-1 1]);
colorbar;
title('右下反射实部');



