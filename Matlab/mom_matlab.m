%--------------------------------------------------------------------------
%                   Majorized Minimization (MoM)                          |
%-------------------------------------------------------------------------+
% Designed to solve the optimization task of the Assisted Dictionary      |
% Learning (ADL) via Majorization Method also known as the Majorized      |
% Minimization algorithm. Using GPU, based on :                           |
% https://github.com/MorCTI/Attom-Assisted-DL/blob/master/Algorithm/MoM.m |
%--------------------------------------------------------------------------
clear;
data = load('data_multi.mat'); % load data from mat file
[a, b, c] = size(data.datamulti);
X = reshape(data.datamulti,a, b*c);
Y = gpuArray(X); % store data into gpu
clearvars -except Y;
K = 10; % Number of components
Tt = 200; % Total number of iterations 
Ts = 20; % Number of iteration to compute the spatial maps
Es = 0;  % (optional parameter) not used in our implementation
Td = 20; % Number of iteration to compute the dictionary
Ed = 0; % (optional parameter) not used in our implementation
lambda =0.5; % The ë of the problem 
ccl=1; % Value of the normalization of each atom 
% Initialization
tic;
[T,N] = size(Y);
lambda = lambda*sqrt(norm(Y,'fro')/(T*N));%Normalization of the parameter ë
D = randn(T,K,'gpuArray');
S = randn(K,N,'gpuArray');
I = gpuArray(diag(ones(1,K)));
fprintf('Initial error : %.6f \n',sqrt(norm(Y-D*S,'fro')/(T*N)));
for i=1:Tt
    % Update Coefficients
    Dux = D'*D;
    cS  = real(max(sqrt(complex(eig(Dux.'*Dux)))));  
    DY = D'*Y/cS;
    Aq = I-Dux/cS;
    Err = 1;
	t = 1;
    bound = 0.5*lambda/cS;
    while (t<=Ts && Err>Es)
		A = DY+Aq*S;
        A = wthresh(A,'s', bound);
		S = A;
		t = t+1;
    end
    % Update Dictionary
    Sux = S*S';
    cD  = real(max(sqrt(complex(eig(Sux.'*Sux)))));
    YS = Y*S'/cD;
    Bq = I-Sux/cD;
	Err = 1;
	t = 1;    
    while (t<=Td && Err > Ed)
        B = YS + D*Bq;
        Kv = sqrt(sum(B.^2));        
        Kv(Kv < ccl) = 1;
        B = bsxfun(@rdivide,B,Kv);
		D = B;
		t = t+1;
    end 
    % Error Computation
    fprintf('Iteration %d : %.6f \n',i,sqrt(norm(Y-D*S,'fro')/(T*N)));

end
disp(['mom cpu time:   ' num2str(toc)]);