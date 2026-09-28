function y_int=fenbujifen(t1,s1,a1,x11,x12)
% y_int=(x12-x11).*((t1==0)&(s1==0))+(exp(t1*a1)/2*(x12-x11)+(exp(t1*x12)+t1*(x12-a1)-exp(t1*x11+t1*(x11-a1)))/4/t1).*(((t1^2+s1^2)==0)&(t1~=0))+(1/(t1^2+s1^2)*exp(t1*x12)*(t1*cos(s1*(x12-a1))+s1*sin(s1*(x12-a1)))-1/(t1^2+s1^2)*exp(t1*x11)*(t1*cos(s1*(x11-a1))+s1*sin(s1*(x11-a1)))).*((t1^2+s1^2)~=0);
%  y_int=(exp(t1*a1)/2*(x12-x11)+(exp(t1*x12)+t1*(x12-a1)-exp(t1*x11+t1*(x11-a1)))/4/t1).*(((t1^2+s1^2)==0)&(t1~=0));
%  y_int= (1/(t1^2+s1^2)*exp(t1*x12)*(t1*cos(s1*(x12-a1))+s1*sin(s1*(x12-a1)))-1/(t1^2+s1^2)*exp(t1*x11)*(t1*cos(s1*(x11-a1))+s1*sin(s1*(x11-a1)))).*((t1^2+s1^2)~=0);
% y_int=(((t1^2+s1^2)==0)&(t1~=0));
%  y_int=(exp(t1*a1)/2*(x12-x11)+(exp(t1*x12)+t1*(x12-a1)-exp(t1*x11+t1*(x11-a1)))/4/t1);
if((t1==0)&&(s1==0))
    y_int=(x12-x11);
elseif(((t1^2+s1^2)==0)&&(t1~=0))
    y_int= (exp(t1*a1)/2*(x12-x11)+(exp(t1*x12)+t1*(x12-a1)-exp(t1*x11+t1*(x11-a1)))/4/t1);
else
     y_int=  (1/(t1^2+s1^2)*exp(t1*x12)*(t1*cos(s1*(x12-a1))+s1*sin(s1*(x12-a1)))-1/(t1^2+s1^2)*exp(t1*x11)*(t1*cos(s1*(x11-a1))+s1*sin(s1*(x11-a1))));
end
end