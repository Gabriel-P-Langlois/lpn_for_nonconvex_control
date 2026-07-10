% Script for the numerical example in paper 1.
% Script written by Gabriel Provencher Langlois

%%% Parameters
x = -5.0:0.1:5.0;
n = length(x);                      
lambda = 2;
t = 1.25;
epsilon = [0.025,0.1,0.25,0.5,1.0];

%%% Numbers and vector quantities
u_pm = zeros(1,n);
v_plus  = (x + t*lambda);
v_minus = (-x + t*lambda);
n_eps = length(epsilon);

%%% Compute u_map and plot u_map vs x
u_map = max(0,x-t*lambda) + min(0,x+t*lambda);
plot(x,u_map,'linewidth',1,'DisplayName','{\bfu}_{map}({\bfx},t)')
hold

%%% Compute u_pm for different values of epsilon and plot them
for k=1:1:n_eps
    ratio_plus  = erfcx(v_plus./sqrt(2*t*epsilon(k)))./(erfcx(v_plus./sqrt(2*t*epsilon(k))) + erfcx(v_minus./sqrt(2*t*epsilon(k))));
    ratio_minus = erfcx(v_minus./sqrt(2*t*epsilon(k)))./(erfcx(v_plus./sqrt(2*t*epsilon(k))) + erfcx(v_minus./sqrt(2*t*epsilon(k))));
    u_pm = v_plus.*ratio_plus - v_minus.*ratio_minus;
    %for i=1:1:n
    %    u_pm(i) = v_plus(i)*ratio_plus(i) - v_minus(i)*ratio_minus(i);
    %end
    plot(x,u_pm,'linewidth',1,'DisplayName',['{\bfu}_{pm}({\bfx},t;\epsilon) with \epsilon = ',num2str(epsilon(k))])
end

%%% Edit the plot further.
xlabel('{\bfx}')
lgd = legend('Location','northwest','Interpreter','tex');
lgd.NumColumns = 2;
grid on
