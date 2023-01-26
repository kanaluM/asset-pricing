function projectGan

BigT=10; %number of time steps
BigN=5;  %number of firms
BigD=2;  %number of moment conditions = number of neurons in 
         %third layer of maximization network
n2=2;    %number of neurons in second layer of both networks

%%%% DATA %%%%%%%%%%

% % market return - identical per firm
x1=[0.02	0.02	0.02	0.02	0.02;
0.0195	0.0195	0.0195	0.0195	0.0195;
0.019	0.019	0.019	0.019	0.019;
0.0185	0.0185	0.0185	0.0185	0.0185;
0.018	0.018	0.018	0.018	0.018;
0.0175	0.0175	0.0175	0.0175	0.0175;
0.017	0.017	0.017	0.017	0.017;
0.0165	0.0165	0.0165	0.0165	0.0165;
0.016	0.016	0.016	0.016	0.016;
0.0155	0.0155	0.0155	0.0155	0.0155];

% % weather - differs per firm
x2=[72	77	88	71	44;
59	66	101	62	78;
65	70	79	38	33;
44	47	112	25	22;
40	49	88	54	12;
51	55	86	50	45;
44	47	55	53	24;
39	44	69	43	12;
68	72	65	70	74;
71	75	74	70	72];

Rexcess=[0.03	0.031	0.032	0.035	0.04;
0.01	0.0305	0.0315	0.0325	0.0255;
-0.01	0.03	0.031	0.0144	0.07;
0.035	0.0295	0.0144	0.0315	0.06;
0.028	0.029	0.03	0.035	0.022;
0.015	0.0285	0.0295	0.0305	0.0315;
0.07	0.028	0.0144	0.03	0.2;
0.0265	0.0275	0.0285	0.0295	0.08;
0.026	0.027	0.028	0.029	0.05;
0.0144	0.0265	0.0144	0.035	0.049];

% Normalizing step, so that x1 and x2 are close to one
% x1=x1/0.02;
% x2=x2/72; % it is always 72 in LA

% Initialize weights and biases
rng(5000);

%start with Wx+b = approx. zero

W2o=0.5*randn(2,2); 
W3o=randn(1,2);
b2o=randn(2,1); % column vector
b3o=randn;
W2g=0.5*randn(2,2);
W3g=0.5*randn(2,2); 
b2g=0.5*randn(2,1); 
b3g=0.5*randn(2,1); 

% W3o=[1 1];
% W2o=[1 0; 0 1];
% b3o=0; 
% b2o=[0;0];
% W2g=[1 0; 0 1];
% W3g=[1 0; 0 1];
% b2g=[0;0];
% b3g=[0;0];

% added AR(1) parameters
epsilon = [1; 1];


s_m = [0.3; 0.3; 0.3; 0.3; 0.3; 0.3];
c_m = [0.8; 0.8];
phi_m = [0.1; 0.1];
% x1 = zeros(10, 5);
% x2 = zeros(10,5);
% for col=1:5
%     x1(1,col)=1;
%     x2(1,col)=1;
% end
% for row=2:10
%     for col=1:5
%         x1(row,col)=c_m(1)+phi_m(1)*x1(row-1,col)+s_m(1)*randn;
%         x2(row,col)=c_m(2)+phi_m(2)*x2(row-1,col)+s_m(1+col)*randn;
%     end
% end
% x1
% x2

for e=1:6
   s_m(e)=0.5;
end
c_m(1) = 0.5;
c_m(2) = 0.5;
phi_m(1) = 0.3;
phi_m(2) = 0.3;

epsilon1 = zeros(BigT, BigN);
epsilon2 = zeros(BigT, BigN);
for i=1:BigN
    epsilon1(1,i)=x1(1,i);
    epsilon2(1,i)=x2(1,i);
    for t=1:BigT-1
        epsilon1(t+1,i)=(x1(t+1,i)-phi_m(1)*x1(t,i)-c_m(1))/s_m(1);
        epsilon2(t+1,i)=(x2(t+1,i)-phi_m(2)*x2(t,i)-c_m(2))/s_m(1+i);
    end
end

eta=2;                   %learning rate
Niter=10;          %number of SG iterations
maxiter=10;         %number of GAN iterations; set to 1 in order to simplify 
savecostmin=zeros(maxiter,1); %will show the progress of minimization
savecostmax=zeros(maxiter,1); %will show the progress of maximization
savecost=zeros(2*maxiter*Niter,1);
storej = zeros(Niter);
stored = zeros(Niter); 

for iter=1:maxiter
    % Dimensions minimization network %%%%%%%%%%%%%%%%%%%%%%%%%%
    %        x1(t,i)          a2o(t,i,1)      a3o(t,i) 
    %        x2(t,i)          a2o(t,i,2)
    %
    % Dim    BigT, BigN,2     BigT,BigN,n2    BigT,BigN
    %               a2o=W2o*x          a3o=W3o*z2  
    %               W2=(n2,2)          w3o=(n2)
    %
    %goal: min s(j,d)^2 over theta_o
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % step 1: initialize g values
    g = zeros(BigT, BigN, 2);
    for t=1:BigT
        for j=1:BigN
            g(t,j,:)=fun_g(t,j,W2g,W3g,b2g,b3g);
        end
    end
%     disp("g after step 1")
%     g(:,:,1)
%     g(:,:,2)

    % step 2
    % calculate gradient with respect to omega and adjust parameters
    for counter=1:Niter
%        for learningcounter=1:10
%            eta=0.0001+(2^learningcounter-2)/1000;
            %forward pass
            a2o = zeros(BigT, BigN, 2);
            a3o = zeros(BigT, BigN);
            for t=1:BigT %calculate omega(counter) 
                for i=1:BigN
                    epsilon = [epsilon1(t,i); epsilon2(t,i)];
                    a2otemp=activate(epsilon,W2o,b2o);
                    a2o(t,i,:)=a2otemp;
                    a3o(t,i)=activate(a2otemp,W3o,b3o);
                    omega(t,i)=a3o(t,i);
                end
            end
%             disp("omega before SG")
%             omega
            newcost=0;
            for j=1:BigN
                for d=1:BigD
                    newcost=newcost+fun_s_jd(j,d,Rexcess, omega, g)^2;
                end
            end

%             disp("total cost before SG")
%             newcost

            delta3o=a3o.*(1-a3o); %matrix
            delta2o=a2o.*(1-a2o); %tensor
            gradient_omega = zeros(BigT, BigN, 19);
            for t=1:BigT
                for i=1:BigN
                    for Index_n2=1:n2
                        delta2o(t,i,Index_n2)=delta2o(t,i,Index_n2)*(W3o(Index_n2)*delta3o(t,i));
                    end
                    e = [epsilon1(t,i); epsilon2(t,i)];
                    gradient_omega(t,i,1)= delta2o(t,i,1)*e(1); %wrt W2(1,1)
                    gradient_omega(t,i,2)= delta2o(t,i,1)*e(2);%wrt W2(1,2)
                    gradient_omega(t,i,3)= delta2o(t,i,2)*e(1);%wrt W2(2,1)
                    gradient_omega(t,i,4)= delta2o(t,i,2)*e(2);%wrt W2(2,2)
                    gradient_omega(t,i,5)= delta2o(t,i,1);%wrt b2(1)
                    gradient_omega(t,i,6)= delta2o(t,i,2);%wrt b2(2)
                    gradient_omega(t,i,7)= delta3o(t,i)*a2o(t,i,1);%wrt W3(1)
                    gradient_omega(t,i,8)= delta3o(t,i)*a2o(t,i,2);%wrt W3(2)
                    gradient_omega(t,i,9)= delta3o(t,i); %wrt b3
                    if t==1
                        for k=10:19
                            gradient_omega(t,i,k)=0;
                        end
                    else
                        gradient_omega(t,i,10)=delta2o(t,i,1)*(-W2o(1,1)*(x1(t,1)-phi_m(1)*x1(t-1,1)+c_m(1))/s_m(1)^2); %wrt s_m(1)
                        gradient_omega(t,i,10+i)=delta2o(t,i,2)*(-W2o(1,2)*(x1(t,1)-phi_m(2)*x2(t-1,i)+c_m(2))/s_m(1+i)^2); %wrt s_m(2,i)
                        gradient_omega(t,i,16)=delta2o(t,i,1)*(-W2o(2,1)*x1(t-1,1)/s_m(1)); %wrt phi_m(1)
                        gradient_omega(t,i,17)=delta2o(t,i,2)*(-W2o(2,2)*x2(t-1,i)/s_m(1+i)); %wrt phi_m(2)
                        gradient_omega(t,i,18)=delta2o(t,i,1)*W2o(1,1)/s_m(1); %wrt c_(1)
                        gradient_omega(t,i,19)=delta2o(t,i,2)*W2o(2,1)/s_m(1+i); %wrt c_(2)
                    end
                end
            end

%             disp("gradient of omega wrt parameters")
%             gradient_omega(:,:,1)
%             gradient_omega(:,:,2)
%             gradient_omega(:,:,3)
%             gradient_omega(:,:,4)
%             gradient_omega(:,:,5)
%             gradient_omega(:,:,6)
%             gradient_omega(:,:,7)
%             gradient_omega(:,:,8)
%             gradient_omega(:,:,9)
%             gradient_omega(:,:,10)
%             gradient_omega(:,:,11)
%             gradient_omega(:,:,12)
%             gradient_omega(:,:,13)
%             gradient_omega(:,:,14)
%             gradient_omega(:,:,15)
%             gradient_omega(:,:,16)
%             gradient_omega(:,:,17)
%             gradient_omega(:,:,18)
%             gradient_omega(:,:,19)

            j=randi(5); %randomize firm
            storej(counter)=j;
            d=randi(2); %randomize moment condition
            stored(counter)=d;
            gradient_Sjdsq = zeros(19);
            for coefftheta=1:19
                for t=1:BigT
                    Sum_i_grad_om=0;
                    for i=1:BigN
                        Sum_i_grad_om=Sum_i_grad_om + gradient_omega(t,i,coefftheta)*Rexcess(t,i);
                    end
                    gradient_Sjdsq(coefftheta)=gradient_Sjdsq(coefftheta)+Rexcess(t,j)*g(t,j,d)*Sum_i_grad_om;
                end
            end
            gradient_Sjdsq=-fun_s_jd(j,d,Rexcess, omega,g)*gradient_Sjdsq;
            W2o(1,1)=W2o(1,1)-eta*gradient_Sjdsq(1);
            W2o(1,2)=W2o(1,2)-eta*gradient_Sjdsq(2);
            W2o(2,1)=W2o(2,1)-eta*gradient_Sjdsq(3);
            W2o(2,2)=W2o(2,2)-eta*gradient_Sjdsq(4);
            b2o(1)=b2o(1)-eta*gradient_Sjdsq(5);
            b2o(2)=b2o(2)-eta*gradient_Sjdsq(6);
            W3o(1)=W3o(1)-eta*gradient_Sjdsq(7);
            W3o(2)=W3o(2)-eta*gradient_Sjdsq(8);
            b3o=b3o-eta*gradient_Sjdsq(9);
            s_m(1)=s_m(1)-eta*gradient_Sjdsq(10);
            s_m(2)=s_m(2)-eta*gradient_Sjdsq(11);
            s_m(3)=s_m(3)-eta*gradient_Sjdsq(12);
            s_m(4)=s_m(4)-eta*gradient_Sjdsq(13);
            s_m(5)=s_m(5)-eta*gradient_Sjdsq(14);
            s_m(6)=s_m(6)-eta*gradient_Sjdsq(15);
            c_m(1)=c_m(1)-eta*gradient_Sjdsq(16);
            c_m(2)=c_m(2)-eta*gradient_Sjdsq(17);
            phi_m(1)=phi_m(1)-eta*gradient_Sjdsq(18);
            phi_m(2)=phi_m(2)-eta*gradient_Sjdsq(19);
            
            for t=1:BigT %calculate tempomega 
                for i=1:BigN
                    omega(t,i)=fun_omega(t,i,W2o,W3o,b2o,b3o);
                end
            end
%             disp("omega after SG")
%             omega

            %Monitor progress
            Mycost=0;
            for j=1:BigN
                for d=1:BigD
                    Mycost=Mycost+fun_s_jd(j,d,Rexcess, omega, g)^2;
                end
            end
%             disp("total cost after SG")
%             Mycost

%        end %loop on eta%
        savecost(2*maxiter*(iter-1)+counter)=Mycost;
        savecostmin(iter)=Mycost;
    end     %step 3


   
    % step 4
    omega=zeros(BigT, BigN,1);
    for t=1:BigT  %update omega(Niter+1) 
        for i=1:BigN
            epsilon = [epsilon1(t,i); epsilon2(t,i)];
            LOCALa2o=activate(epsilon,W2o,b2o);
            LOCALa3o=activate(LOCALa2o,W3o,b3o);
            omega(t,i)=LOCALa3o;
        end
    end

    %m(t) is the sum of (1-(omega*R)(t,i)) across i
    inner = zeros(1, BigT);
    m = zeros(1, BigT);
    for t=1:BigT
        inner(t)=0;
        for i=1:BigN
            inner(t)=inner(t)+omega(t,i)*Rexcess(t,i);
        end
        m(t)=1-inner(t);
    end

%     disp("m(t) at the end of step 4")
%     m

    % step 5
    % Dimensions maximization network %%%%%%%%%%%%%%%%%%%%%%%%%%
    % j is FIXED, so x1(t) means x1(t,j)
    %        x1(t)          a2g(t,1)        a3g(t,1) 
    %        x2(t)          a2o(t,2)        a3g(t,2)
    %
    % Dim    BigT,2         BigT,n2         BigT,BigD
    %               z2g=W2g*x+b2g      z3g=W3o*a2g+b3g  
    %               W2g=(n2,2)         w3g=(BigD,n2) 
    %
    %goal: min s(j,d)^2 over theta_g
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for counter=1:Niter
%        for learningcounter=1:10
 %           newcost=10000000;
 %           eta=0.0001+(2^learningcounter-2)/1000;
            fixedj=randi(5); %randomize firm
            storej(counter)=fixedj;
            d=randi(2); %randomize moment condition
            stored(counter)=d;
        %forward pass
            a2g = zeros(BigT, n2);
            a3g = zeros(BigT, BigD);
            for t=1:BigT
                x=[x1(t,fixedj);x2(t,fixedj)];
                a2gtemp=activate(x,W2g,b2g); %vector (n2)
                a2g(t,:)=a2gtemp; %matrix (BigT,n2)
                a3g(t,:)=activate(a2gtemp,W3g,b3g); %matrix dim (BigT,BigD)
            end
            delta3g=a3g.*(1-a3g); %matrix (BigT, BigD)
            delta2g=a2g.*(1-a2g); %matrix (BigT, n2)

            delta3g_chosend = zeros(Niter, BigT);
            gradient_g = zeros(Niter, BigT);
            for t=1:BigT
                delta3g_chosend(t)=delta3g(t,d); %vector (BigT)
                W3g_chosend(:)=W3g(d,:);         %vector (n2)
                delta2g_chosend = zeros(Niter, BigT, n2);
                for Index_n2=1:n2
                    delta2g_chosend(t,Index_n2)= ...
                    delta2g(t,Index_n2)*W3g_chosend(Index_n2)*delta3g_chosend(t);
                end
                % we pick only one moment
                gradient_g(t,1)= delta2g_chosend(t,1)*x1(t, fixedj);   %dg_d/dW^2(1,1) 
                gradient_g(t,2)= delta2g_chosend(t,1)*x2(t,fixedj);   %dg_d/dW^2(1,2)
                gradient_g(t,3)= delta2g_chosend(t,2)*x1(t,fixedj);   %dg_d/dW^2(2,1)
                gradient_g(t,4)= delta2g_chosend(t,2)*x2(t,fixedj);   %dg_d/dW^2(2,2)
                gradient_g(t,5)= delta2g_chosend(t,1);         %dg_d/db^2(1)
                gradient_g(t,6)= delta2g_chosend(t,2);         %dg_d/db^2(2)
                gradient_g(t,7)= delta3g_chosend(t)*a2g(t,1);  %dg_d/dW^3(d,1)                  
                gradient_g(t,8)= delta3g_chosend(t)*a2g(t,2);  %dg_d/dW^3(d,2)
                gradient_g(t,9)= delta3g_chosend(t);           %dg_d/db^3(d)   
            end
            for coefftheta=1:9
                gradient_Sjdsq(coefftheta)=0;
                for t=1:BigT
                    gradient_Sjdsq(coefftheta)=gradient_Sjdsq(coefftheta)+ ...
                        m(t)*Rexcess(t,fixedj)*gradient_g(t,coefftheta);
                end
            end
            gradient_Sjdsq=fun_s_jd(j,d, Rexcess, omega, g)*gradient_Sjdsq;       
            W2g(1,1)=W2g(1,1)+eta*gradient_Sjdsq(1);
            W2g(1,2)=W2g(1,2)+eta*gradient_Sjdsq(2);
            W2g(2,1)=W2g(2,1)+eta*gradient_Sjdsq(3);
            W2g(2,2)=W2g(2,2)+eta*gradient_Sjdsq(4);
            b2g(1)=b2g(1)+eta*gradient_Sjdsq(5);
            b2g(2)=b2g(2)+eta*gradient_Sjdsq(6);
            W3g(d,1)=W3g(1,1)+eta*gradient_Sjdsq(7);
            W3g(d,2)=W3g(1,2)+eta*gradient_Sjdsq(8);
            b3g(d)=  b3g(d)+eta*gradient_Sjdsq(9);
            %Monitor progress
            for t=1:BigT %calculate tempomega 
                for i=1:BigN
                    g(t,j,:)=fun_g(t,j,W2g, W3g,b2g,b3g);
                end
            end
%             disp("g after SF")
%             g(t,j,1)

            %Monitor progress
            Mycost=0;
            for j=1:BigN
                for d=1:BigD
                    Mycost=Mycost+fun_s_jd(j,d,Rexcess, omega, g)^2;
                end
            end
%             disp("total cost at end of step 5")
%             Mycost
%            if Mycost<newcost
%               newcost=Mycost;
%               W2g=MyW2g; b2g=Myb2g; W3g=MyW3g; b3g=Myb3g;
%            end
%        end %eta
         savecost(2*maxiter*(iter-1)+Niter+counter)=Mycost;
         savecostmax(iter)=Mycost;    
    end % counter
%    plot(savecostmin)
%     hold on
%    plot(savecostmax)
%     storej;
%     stored;
end

% savecostmax
% savecostmin

% c_m
% phi_m
% s_m
plot(savecost, '-')
% plot(savecostmin,'-')
% hold on
% plot(savecostmax,'--')
title('Cost function vs GAN iteration (LSTM)')
xlabel('iteration') 
ylabel('Cost') 
% legend({'min cost', 'max cost'},'Location','southeast')

function s_j_val=fun_s_j(j,Rexcess, omega, g)
    s_j_val=0;
    for dd=1:BigD        
        s_j_val=s_j_val+fun_s_jd(j,dd,Rexcess, omega, g);
    end 
end

function s_jd_val=fun_s_jd(j,d,Rexcess, omega, g)
    s_jd_val=0;
    for tt=1:BigT
        portfret=0;
         for ii=1:BigN
            portfret=portfret+omega(tt,ii)*Rexcess(tt,ii);
        end
        s_jd_val=s_jd_val+(1-portfret)*Rexcess(tt,j)*g(tt,j,d);
    end
end %of nested function

function omega_ti_val=fun_omega(t,i,W2o,W3o,b2o,b3o)
    epsilon = [epsilon1(t,i); epsilon2(t,i)];
    LOCALa2o=activate(epsilon,W2o,b2o);
    LOCALa3o=activate(LOCALa2o,W3o,b3o);
    omega_ti_val=LOCALa3o;
end %of nested function

function g_val=fun_g(t,j,W2g,W3g,b2g,b3g)
    x = [x1(t,j); x2(t,j)];
    LOCALa2g=activate(x,W2g,b2g);
    LOCALa3g=activate(LOCALa2g,W3g,b3g);
    g_val=LOCALa3g;
end

function y=activate(x,W,b)
%evaluates sigmoid function
y=1./(1+exp(-(W*x+b)));
end

end