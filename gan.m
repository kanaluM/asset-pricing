function testData
BigT=10; %number of time steps
BigN=5;  %number of firms
BigD=2;  %number of moment conditions = number of neurons in 
         %third layer of maximization network
n2=2;    %number of neurons in second layer of both networks
%%%% DATA %%%%%%%%%%
x1=[0.02 0.02 0.02 0.02 0.02;
0.0195 0.0195 0.0195 0.0195 0.0195;
0.019 0.019 0.019 0.019 0.019;
0.0185 0.0185 0.0185 0.0185 0.0185;
0.018 0.018 0.018 0.018 0.018;
0.0175 0.0175 0.0175 0.0175 0.0175;
0.017 0.017 0.017 0.017 0.017;
0.0165 0.0165 0.0165 0.0165 0.0165;
0.016 0.016 0.016 0.016 0.016;
0.0155 0.0155 0.0155 0.0155 0.0155]; %market return

x2=[72 77 88 71 44;
59 66 101 62 78;
65 70 79 38 33;
44 47 112 25 22;
40 49 88 54 12;
51 55 86 50 45;
44 47 55 53 24;
39 44 69 43 12;
68 72 65 70 74;
71 75 74 70 72]; %weather

Rexcess=[0.03 0.031 0.032 0.035 0.04;
0.01 0.0305 0.0315 0.0325 0.0255;
-0.01 0.03 0.031 0.0144 0.07;
0.035 0.0295 0.0144 0.0315 0.06;
0.028 0.029 0.03 0.035 0.022;
0.015 0.0285 0.0295 0.0305 0.0315;
0.07 0.028 0.0144 0.03 0.2;
0.0265 0.0275 0.0285 0.0295 0.08;
0.026 0.027 0.028 0.029 0.05;
0.0144 0.0265 0.0144 0.035 0.049];

% x1 = csvread('MarketReturn.csv');
% x2 = csvread('Weather.csv');
% Rexcess= csvread('Return.csv');

%Normalizing step, so that x1 and x2 are close to one
x1=x1/0.02;
x2=x2/72; %did you see LA story? It is always 72 in LA

% Initialize weights and biases
rng(5000);
%start with Wx+b = approx. zero
% W2o=0.5*randn(2,2);
% W3o=[1 1];
% x=[x1(1,1); x2(1,1)];
% b2o=-W2o*x;
% b3o=0; 
% W2g=0.5*randn(2,2);
% W3g=0.5*randn(2,2);
% b2g=0.5*randn(2,1);
% b3g=0.5*randn(2,1);
W2o=[1 0;0 1];
W3o=[1 1];
W2o=[1 0; 0 1];
x=[x1(1,1); x2(1,1)];
b3o=0; 
b2o=[0;0];
W2g=[1 0; 0 1];
W3g=[1 0; 0 1];
b2g=[0;0];
b3g=[0;0];

eta=2;             %learning rate
Niter=1;          %number of SG iterations
maxiter=1;         %number of GAN iterations; set to 1 in order to simplify 
savecostmin=zeros(Niter,1); %will show the prohress of minimization
savecostmax=zeros(Niter,1); %will show the progress of maximization
savecost=zeros(Niter,2);
omega=zeros(BigT, BigN,1);
for iter=1:maxiter

    % step 1
    g = zeros(BigT, BigN, 2);
    for t=1:BigT
        for j=1:BigN
            x=[x1(t,j); x2(t,j)];
            g(t,j,:) = fun_g(x,W2g,W3g,b2g,b3g);
        end
    end
    g

    % step 2
    a2o = zeros(BigT, BigN, 2);
    a3o = zeros(BigT, BigN);
    storej = zeros(Niter);
    stored = zeros(Niter);
    gradient_Sjdsq = zeros(9);
    for counter=1:Niter
        for t=1:BigT %calculate omega(counter) 
            for i=1:BigN
                x=[x1(t,i);x2(t,i)];
                a2otemp=activate(x,W2o,b2o);
                a2o(t,i,:)=a2otemp;
                a3o(t,i)=activate(a2otemp,W3o,b3o);
                omega(t,i)=a3o(t,i);
            end
        end
        delta3o=a3o.*(1-a3o); 
        delta2o=a2o.*(1-a2o);
        gradient_omega = zeros(BigT, BigN, 9);
        for t=1:BigT
            for i=1:BigN
                for Index_n2=1:n2
                    delta2o(t,i,Index_n2)=delta2o(t,i,Index_n2)*(W3o(Index_n2)*delta3o(t,i));
                end
                gradient_omega(t,i,1)= delta2o(t,i,1)*x1(t,i); %wrt W2(1,1)
                gradient_omega(t,i,2)= delta2o(t,i,1)*x2(t,i);%wrt W2(1,2)
                gradient_omega(t,i,3)= delta2o(t,i,2)*x1(t,i);%wrt W2(2,1)
                gradient_omega(t,i,4)= delta2o(t,i,2)*x2(t,i);%wrt W2(2,2)
                gradient_omega(t,i,5)= delta2o(t,i,1);%wrt b2(1)
                gradient_omega(t,i,6)= delta2o(t,i,2);%wrt b2(2)
                gradient_omega(t,i,7)= delta3o(t,i)*a2o(t,i,1);%wrt W3(1)
                gradient_omega(t,i,8)= delta3o(t,i)*a2o(t,i,2); %wrt W3(2)
                gradient_omega(t,i,9)= delta3o(t,i); %wrt b3
            end
        end

        % j=randi(5); %randomize firm
        j = 1;
        storej(counter)=j;
        % d=randi(2); %randomize moment condition
        d = 1;
        stored(counter)=d;

        for coefftheta=1:9
            gradient_Sjdsq(coefftheta)=0;
            for t=1:BigT
                Sum_i_grad_om=0;
                for i=1:BigN
                    Sum_i_grad_om=Sum_i_grad_om +gradient_omega(t,i,coefftheta)*Rexcess(t,i);
                end
                gradient_Sjdsq(coefftheta)=gradient_Sjdsq(coefftheta)+Rexcess(t,j)*g(t,j,d)*Sum_i_grad_om;
            end
        end
        gradient_Sjdsq=-fun_s_jd(Rexcess, x,j,d)*gradient_Sjdsq;
        W2o(1,1)=W2o(1,1)-eta*gradient_Sjdsq(1);
        W2o(1,2)=W2o(1,2)-eta*gradient_Sjdsq(2);
        W2o(2,1)=W2o(2,1)-eta*gradient_Sjdsq(3);
        W2o(2,2)=W2o(2,2)-eta*gradient_Sjdsq(4);
        b2o(1)=b2o(1)-eta*gradient_Sjdsq(5);
        b2o(2)=b2o(2)-eta*gradient_Sjdsq(6);
        W3o(1)=W3o(1)-eta*gradient_Sjdsq(7);
        W3o(2)=W3o(2)-eta*gradient_Sjdsq(8);
        b3o = b3o-eta*gradient_Sjdsq(9);

        % monitor progress
        newcost=0;
        for j=1:BigN
            newcost=newcost+fun_s_j(Rexcess, x,j);
        end
        savecostmin(counter)=newcost;
    end     
    
    
    % step 4
    for t=1:BigT  % update omega(Niter+1) 
        for i=1:BigN
            x=[x1(t,i);x2(t,i)];
            a2otemp=activate(x,W2o,b2o);
            a2o(t,i,:)=a2otemp;
            a3o(t,i)=activate(a2otemp,W3o,b3o);
            omega(t,i)=a3o(t,i);
        end
    end
       
    % precompute inner sum with w(x, theta)
    inner = zeros(BigT);
    m = zeros(BigT);
    for t=1:BigT
        inner(t)=0;
        for i=1:BigN
            inner(t)=inner(t)+omega(t,i)*Rexcess(t,i);
        end
        m(t)=1-inner(t);
    end

    % step 5
    gradient_g = zeros(BigT, 9);
    a2g = zeros(BigT, 2);
    a3g = zeros(t, BigD);
    delta3g_chosend = zeros(BigT);
    delta2g_chosend = zeros(BigT,n2);
    for counter=1:Niter
        % fixedj=randi(5); % randomize firm
        fixedj = 1;
        storej(counter)=fixedj;
        % d=randi(2); % randomize moment condition
        d = 1;
        stored(counter)=d;
        %forward pass
        for t=1:BigT
            x=[x1(t,fixedj);x2(t,fixedj)];
            a2gtemp=activate(x,W2g,b2g); %vector (n2)
            a2g(t,:)=a2gtemp; %matrix (BigT,n2)
            a3g(t,:)=activate(a2gtemp,W3g,b3g); %matrix dim (BigT,BigD)
        end
        delta3g=a3g.*(1-a3g); %matrix (BigT, BigD)
        delta2g=a2g.*(1-a2g); %matrix (BigT, n2)
        for t=1:BigT
            delta3g_chosend(t)=delta3g(t,d); %vector (BigT)
            W3g_chosend(:)=W3g(d,:);         %vector (n2)
            for Index_n2=1:n2
                delta2g_chosend(t,Index_n2)= ...
                delta2g(t,Index_n2)*W3g_chosend(Index_n2)*delta3g_chosend(t);
            end

            % it's possible they give the same answer for the test cases?
            % my version of the gradient
%             gradient_g(t,1)= delta2g(t,1) * x1(t,1);  % dg_d/dW^2(1,1) 
%             gradient_g(t,2)= delta2g(t,1) * x2(t,2);  % dg_d/dW^2(1,2)
%             gradient_g(t,3)= delta2g(t,2) * x1(t,1);  % dg_d/dW^2(2,1)
%             gradient_g(t,4)= delta2g(t,2) * x2(t,2);  % dg_d/dW^2(2,2)
%             gradient_g(t,5)= delta2g(t,1);            % dg_d/db^2(1)
%             gradient_g(t,6)= delta2g(t,2);            % dg_d/db^2(2)
%             gradient_g(t,7)= delta3g(d) * a2g(t,1);   % dg_d/dW^3(d,1)                  
%             gradient_g(t,8)= delta3g(d) * a2g(t,2);   % dg_d/dW^3(d,2)
%             gradient_g(t,9)= delta3g(d);              % dg_d/db^3(d)

            % Professor's version of the gradient
            gradient_g(t,1)= delta2g_chosend(t,1) * x1(t,fixedj);  % dg_d/dW^2(1,1) 
            gradient_g(t,2)= delta2g_chosend(t,1) * x2(t,fixedj);  % dg_d/dW^2(1,2)
            gradient_g(t,3)= delta2g_chosend(t,2) * x1(t,fixedj);  % dg_d/dW^2(2,1)
            gradient_g(t,4)= delta2g_chosend(t,2) * x2(t,fixedj);  % dg_d/dW^2(2,2)
            gradient_g(t,5)= delta2g_chosend(t,1);            % dg_d/db^2(1)
            gradient_g(t,6)= delta2g_chosend(t,2);            % dg_d/db^2(2)
            gradient_g(t,7)= delta3g_chosend(t) * a2g(t,1);   % dg_d/dW^3(d,1)                  
            gradient_g(t,8)= delta3g_chosend(t) * a2g(t,2);   % dg_d/dW^3(d,2)
            gradient_g(t,9)= delta3g_chosend(t);              % dg_d/db^3(d)   

        end

        gradient_Sjdsq = zeros(9);
        for coefftheta=1:9
            for t=1:BigT
                gradient_Sjdsq(coefftheta)=gradient_Sjdsq(coefftheta)+ ...
                    m(t)*Rexcess(t,j)*gradient(t,coefftheta);
            end
        end

        % update parameters
        gradient_Sjdsq=fun_s_jd(Rexcess,x,fixedj,d)*gradient_Sjdsq;
        MyW2g(1,1) = W2g(1,1)+eta*gradient_Sjdsq(1);
        MyW2g(1,2) = W2g(1,2)+eta*gradient_Sjdsq(2);
        MyW2g(2,1) = W2g(2,1)+eta*gradient_Sjdsq(3);
        MyW2g(2,2) = W2g(2,2)+eta*gradient_Sjdsq(4);
        Myb2g(1) = b2g(1)+eta*gradient_Sjdsq(5);
        Myb2g(2) = b2g(2)+eta*gradient_Sjdsq(6);
        MyW3g(d,1) = W3g(1,1)+eta*gradient_Sjdsq(7);
        MyW3g(d,2) = W3g(1,2)+eta*gradient_Sjdsq(8);
        Myb3g(d) = b3g(d)+eta*gradient_Sjdsq(9);

        %Monitor progress
        for t=1:BigT
            for j=1:BigN
                x=[x1(t,j);x2(t,j)];
                g(t,j,:)=fun_g(x,W2g,W3g,b2g,b3g);
            end
        end
        newcost=0;
        for j=1:BigN
            newcost=newcost+fun_s_j(Rexcess, x,j);
        end
        savecostmax(counter)=newcost;    
    end 
%     savecost = [savecostmin savecostmax];
%     plot(savecostmin)
%     hold on
%     plot(savecostmax);
%     storej;
%     stored;
end
% plot(savecostmin, '-')
% hold on
% plot(savecostmax, '-');

function s_j_val=fun_s_j(Rexcess,x,j)
    s_j_val=0;
    for dd=1:BigD
        s_j_val=s_j_val+fun_s_jd(Rexcess, x, j,dd);
    end
end

function s_jd_val=fun_s_jd(Rexcess,x,j,d)
    s_jd_val=0;
    for tt=1:BigT
        portfret=0;
        for ii=1:BigN
            omega_ti=a3o(tt,ii);
            portfret=portfret+omega_ti*Rexcess(tt,ii);
        end
        s_jd_val=s_jd_val+(1-portfret)*Rexcess(tt,j)*g(tt,j,d);
    end
end

function omega_ti_val=fun_omega(x,W2o, W3o,b2o,b3o)
    a2o=activate(x,W2o,b2o);
    a3o=activate(a2o,W3o,b3o);
    omega_ti_val=a3o;
end %of nested function

function g_val=fun_g(x,W2g, W3g,b2g,b3g)
    LOCALa2g=activate(x,W2g,b2g);
    LOCALa3g=activate(LOCALa2g,W3g,b3g);
    g_val=LOCALa3g;
end

function y=activate(x,W,b)
%evaluates sigmoid function
    y=1./(1+exp(-(W*x+b)));
end
end