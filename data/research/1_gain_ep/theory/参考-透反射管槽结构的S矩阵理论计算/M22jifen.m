
function y_int=M22jifen(a1,a2,x11,x12)

y_int=(x12-x11)/2.*((a1==a2)&(a1~=0)&(a2~=0))+(0).*(a1~=a2)+(x12-x11).*((a2==0)&(a1==0));

end