clear
clc
warning off;

path = pwd;
addpath '.\funs'
% addpath(genpath(path));
Dataname = 'Yale_jie';
load(Dataname);
par = [1e5,1e4,1e3,1e2,1e1,1,1e-1,1e-2,1e-3,1e-4,1e-5];
n = length(Y);
% par = [1e1];
for i = 1:length(X)
    A_v = X{i};%Œﬁ‘Î…˘
%     A_v = X{i}+ (sqrt(mean(std(X{i}))).*randn(size(X{i}))); %º”»Î∏ﬂÀπ‘Î…˘
%     A_v = imnoise(X{i},"salt & pepper",0.1); %Ω∑—Œ‘Î…˘
%         A_v = (X{i}-mean(X{i}))./(max(X{i})-min(X{i})+eps);
    X{i}  =A_v./repmat(sqrt(sum(A_v.^2)+eps),[size(A_v,1) 1]);
    X{i} = (X{i}-mean(X{i}))./(max(X{i})-min(X{i})+eps);
end
% par = [1e2];
anchor_rate = 1;
[CS,centers] = gen_anchor_similarity(X,anchor_rate);
anchor_num = fix(n*anchor_rate);
for dd = 1:length(par)
    iter =1;
    for runtime = 1:iter 
        load(Dataname);
        lambda = par(dd);
        for name = 1
            numclass = length(unique(Y));
            A=zeros(n,anchor_num,length(X));
            %         KH = A;
            for i = 1:length(X)
                A_v = CS{i};
%                             A_v  =A_v./repmat(sqrt(sum(A_v.^2)+eps),[size(A_v,1) 1]);
                %             A(:,:,i) = constructW(A_v,options2) + eye(size(A_v,1));
                KH(:,:,i) = exp(1 - A_v);
            end
            
            options.seuildiffsigma=1e-3;        % stopping criterion for weight variation
            %------------------------------------------------------
            % Setting some numerical parameters
            %------------------------------------------------------
            options.goldensearch_deltmax=1e-1; % initial precision of golden section search
            options.numericalprecision=1e-16;   % numerical precision weights below this value
            % are set to zero
            %------------------------------------------------------
            % some algorithms paramaters
            %------------------------------------------------------
            options.firstbasevariable='first'; % tie breaking method for choosing the base
            % variable in the reduced gradient method
            options.nbitermax=50;             % maximal number of iteration
            options.seuil=0;                   % forcing to zero weights lower than this
            options.seuilitermax=10;           % value, for iterations lower than this one
            options.miniter=0;                 % minimal number of iterations
            options.threshold = 1e-3;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [S,Sigma,obj] = GOAL_MVC(KH, options,lambda);
               S1 = (S + S') / 2;
            D = diag(1 ./ sqrt(sum(S1)));
            L =  D * S1 * D;
            [H,~] = eigs(L, numclass, 'LA');
            res= myNMIACC(H,Y,numclass);

            disp(res);
%             dlmwrite(Dataname+".txt",[lambda,res(1) res(2) res(3)],'-append','delimiter','\t','newline','pc');
        end
    end
end