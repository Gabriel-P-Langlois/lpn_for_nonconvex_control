% Script for my research project with Jerome
% Verify equation (3.1) of my document for one value of x.

lhs = integral(@(x) myfun1(x),-inf,inf);
x = -2:0.1:2;
vect = zeros(1,length(x));
for j=1:1:length(x)
    vect(j) = (x(j)^2)/2 + myfun2(x(j));
end
plot(x,vect)

function y = myfun1(x)
y = exp(-abs(x) - 0.5*(x.^2))/sqrt(2*pi);
end

function z = myfun2(mu)
% x     : scalar
% mu    : scalar
my_int = @(x) abs(x).*exp(-0.5.*((x-mu).^2))/sqrt(2*pi);
z = integral(my_int,-inf,inf);
end
