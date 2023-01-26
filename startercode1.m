function GAN_Project2_20221121
BigT=10; %number of time steps
BigN=5;  %number of firms
BigD=2;  %number of moment conditions = number of neurons in 
         %third layer of maximization network
n2=2;    %number of neurons in second layer of both networks
%%%% DATA %%%%%%%%%%
x1=[0.02	0.02	0.02	0.02	0.02;
0.0195	0.0195	0.0195	0.0195	0.0195;
0.019	0.019	0.019	0.019	0.019;
0.0185	0.0185	0.0185	0.0185	0.0185;
0.018	0.018	0.018	0.018	0.018;
0.0175	0.0175	0.0175	0.0175	0.0175;
0.017	0.017	0.017	0.017	0.017;
0.0165	0.0165	0.0165	0.0165	0.0165;
0.016	0.016	0.016	0.016	0.016;
0.0155	0.0155	0.0155	0.0155	0.0155]; %market return

x2=[72	77	88	71	44;
59	66	101	62	78;
65	70	79	38	33;
44	47	112	25	22;
40	49	88	54	12;
51	55	86	50	45;
44	47	55	53	24;
39	44	69	43	12;
68	72	65	70	74;
71	75	74	70	72]; %weather

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

%Normalizing step, so that x1 and x2 are close to one
x1=x1/0.02;
x2=x2/72;


% Initialize weights and biases
%rng(5000);
%start with Wx+b = approx. zero
%W2o=0.5*randn(2,2); MyW2o=W2o;
W3o=[1 1];
W2o=[1 0; 0 1];
W3o=[1 1];
b2o=[0;0];
b3o=[0];
c1=2; c2=2;
phi1=0.5;phi2=0.5;
sigma1=1;
sigma2=1;

W2g=[1 0; 0 1];
W3g=[1 0; 0 1];
b2g=[0;0];
b3g=[0;0];
for i=1: BigN
    epsilon1(1,i)=x1(1,i);
    epsilon2(1,i)=x2(1,i);
    for t=1:BigT-1
        epsilon1(t+1,i)=(x1(t+1,i)-phi1*x1(t,i)-c1)/sigma1;
        epsilon2(t+1,i)=(x2(t+1,i)-phi2*x2(t,i)-c2)/sigma2;
    end
end
epsilon2

eta=2;                   %learning rate
Niter=11;          %number of SG iterations
maxiter=11;         %number of GAN iterations; set to 1 in order to simplify 
savecostmin=zeros(Niter,1); %will show the prohress of minimization
savecostmax=zeros(Niter,1); %will show the progress of maximization
savecost=zeros(Niter,2);
omega=zeros(BigT, BigN,1);
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

    % step 1: initializations 
    for t=1:BigT
        for j=1:BigN
            g(t,j,:)=fun_g(t,j,x1,x2,W2g,W3g,b2g,b3g);
        end
    end
    % step 2
    for counter=1:Niter
%        for learningcounter=1:10
%            newcost=10000000;
%            eta=0.0001+(2^learningcounter-2)/1000;
            %forward pass
            for t=1:BigT %calculate omega(counter) 
                for i=1:BigN
                    x=[x1(t,i);x2(t,i)];
                    a2otemp=activate(x,W2o,b2o);
                    a2o(t,i,:)=a2otemp;
                    a3o(t,i)=activate(a2otemp,W3o,b3o);
                    omega(t,i)=a3o(t,i);
                end
            end
            delta3o=a3o.*(1-a3o); %matrix
            delta2o=a2o.*(1-a2o); %tensor
            for t=1:BigT
                for i=1:BigN
                %11/12/2021 changed name of indexing variable from d to Index_n2 and
                %BigB to n2
                    for Index_n2=1:n2
                        delta2o(t,i,Index_n2)=delta2o(t,i,Index_n2)*(W3o(Index_n2)*delta3o(t,i));
                    end
                    gradient_omega(t,i,1)= delta2o(t,i,1)*epsilon1(t,i); %wrt W2(1,1)
                    gradient_omega(t,i,2)= delta2o(t,i,1)*epsilon2(t,i);%wrt W2(1,2)
                    gradient_omega(t,i,3)= delta2o(t,i,2)*epsilon1(t,i);%wrt W2(2,1)
                    gradient_omega(t,i,4)= delta2o(t,i,2)*epsilon2(t,i);%wrt W2(2,2)
                    gradient_omega(t,i,5)= delta2o(t,i,1);%wrt b2(1)
                    gradient_omega(t,i,6)= delta2o(t,i,2);%wrt b2(2)
                    gradient_omega(t,i,7)= delta3o(t,i)*a2o(t,i,1);%wrt W3(1)
                    gradient_omega(t,i,8)= delta3o(t,i)*a2o(t,i,2); %wrt W3(2)
                    gradient_omega(t,i,9)= delta3o(t,i); %wrt b3
                    if t==1
                        for k=10:15
                            gradient_omega(t,i,k)=0;
                        end
                    else
                        gradient_omega(t,i,10)=delta2o(t,i,1)*(-W2o(1,1)*(x1(t,1)-phi1*x1(t-1,1)+c1)/sigma1^2); %do/dsigma1
                        gradient_omega(t,i,11)=delta2o(t,i,2)*(-W2o(1,2)*(x1(t,1)-phi2*x2(t-1,i)+c2)/sigma2^2); %do/dsigma2
                        gradient_omega(t,i,12)=delta2o(t,i,1)*(-W2o(2,1)*x1(t-1,1)/sigma1); %do/dphi1
                        gradient_omega(t,i,13)=delta2o(t,i,2)*(-W2o(2,2)*x2(t-1,i)/sigma2); %do/dphi2
                        gradient_omega(t,i,14)=delta2o(t,i,1)*W2o(1,1)/sigma1; %do/dc1
                        gradient_omega(t,i,15)=delta2o(t,i,2)*W2o(2,1)/sigma2; %do/dc2
                    end
                end
            end
            disp("gradient of omega wrt parameters")
            gradient_omega
            j=randi(5); %randomize firm
            storej(counter)=j;
            d=randi(2); %randomize moment condition
            %j=1; d=2; I used this to test convergence on a fixed s(j,d)
            stored(counter)=d;
            for coefftheta=1:15
                gradient_Sjdsq(coefftheta)=0;
                for t=1:BigT
                    Sum_i_grad_om=0;
                    for i=1:BigN
                     %11/12/2021 add Rexcess(t,i)
                        Sum_i_grad_om=Sum_i_grad_om +gradient_omega(t,i,coefftheta)*Rexcess(t,i);
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
            sigma1=sigma1-eta*gradient_Sjdsq(10);
            sigma2=sigma2-eta*gradient_Sjdsq(11);
            phi1=phi1-eta*gradient_Sjdsq(12);
            phi2=phi2-eta*gradient_Sjdsq(13);
            c1=c1-eta*gradient_Sjdsq(14);
            c2=c2-eta*gradient_Sjdsq(15);

            for t=1:BigT %calculate tempomega 
                for i=1:BigN
                    omega(t,i)=fun_omega(t,i,x1,x2,W2o,W3o,b2o,b3o);
                end
            end
            %Monitor progress
            Mycost=0;
            for j=1:BigN
                Mycost=Mycost+fun_s_j(j,Rexcess, omega, g);
            end
%            if Mycost<newcost
%               newcost=Mycost;
%               W2o=MyW2o; b2o=Myb2o; W3o=MyW3o; b3o=Myb3o;
%            end
 
%        end %loop on eta%
        savecostmin(counter)=Mycost;
    end     %step 3

    storej;
    stored;
   
    %step 4
    for t=1:BigT  %update omega(Niter+1) 
        for i=1:BigN
            x=[x1(t,i);x2(t,i)];
            a2otemp=activate(x,W2o,b2o);
            a2o(t,i,:)=a2otemp;
            a3o(t,i)=activate(a2otemp,W3o,b3o);
            omega(t,i)=a3o(t,i);
        end
    end

    %m(t) is the sum of (1-(omega*R)(t,i)) across i
    for t=1:BigT
        inner(t)=0;
        for i=1:BigN
            inner(t)=inner(t)+omega(t,i)*Rexcess(t,i);
        end
        %11/12/2021 renamed inner(t) into m(t)
        m(t)=1-inner(t);
    end

end

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


function omega_ti_val=fun_omega(t,i,x1,x2,W2o,W3o,b2o,b3o)
    x=[x1(t,i);x2(t,i)];
    LOCALa2o=activate(x,W2o,b2o);
    LOCALa3o=activate(LOCALa2o,W3o,b3o);
    omega_ti_val=LOCALa3o;
end %of nested function

function g_val=fun_g(t,j,x1,x2,W2g, W3g,b2g,b3g)
    x=[x1(t,j);x2(t,j)];
    LOCALa2g=activate(x,W2g,b2g);
    LOCALa3g=activate(LOCALa2g,W3g,b3g);
    g_val=LOCALa3g;
end
function y=activate(x,W,b)
%evaluates sigmoid function
    y=1./(1+exp(-(W*x+b)));
end
end