%  HYDRA
%  Version 1.0.0 --- January 2018
%  Section of Biomedical Image Analysis
%  Department of Radiology
%  University of Pennsylvania
%  Richard Building
%  3700 Hamilton Walk, 7th Floor
%  Philadelphia, PA 19104
%
%  Web:   https://www.med.upenn.edu/sbia/
%  Email: sbia-software at uphs.upenn.edu
%
%  Copyright (c) 2018 University of Pennsylvania. All rights reserved.
%  See https://www.med.upenn.edu/sbia/software-agreement.html or COPYING file.
%
%  Author:
%  Erdem Varol
%  software@cbica.upenn.edu


function [CIDX,ARI] = hydra(varargin)

if nargin==0
    printhelp()
    return
end

if( strcmp(varargin{1},'--help') || isempty(varargin))
    printhelp()
    return;
end

if( strcmp(varargin{1},'-h') || isempty(varargin) )
    printhelp()
    return
end

if( strcmp(varargin{1},'--version') || isempty(varargin) )
    fprintf('Version 1.0.\n')
    return
end

if( strcmp(varargin{1},'-v') || isempty(varargin) )
    fprintf('Version 1.0.\n')
    return
end

if( strcmp(varargin{1},'-u') || isempty(varargin) )
    fprintf(' EXAMPLE USE (in matlab) \n');
    fprintf(' hydra(''-i'',''test.csv'',''-o'',''.'',''-k'',3,''-f'',3) \n');
    fprintf(' EXAMPLE USE (in command line) \n');
    fprintf(' hydra -i test.csv -o . -k 3 -f 3 \n');
    return
end

if( strcmp(varargin{1},'--usage') || isempty(varargin) )
    fprintf(' EXAMPLE USE (in matlab) \n');
    fprintf(' hydra(''-i'',''test.csv'',''-o'',''.'',''-k'',3,''-f'',3) \n');
    fprintf(' EXAMPLE USE (in command line) \n');
    fprintf(' hydra -i test.csv -o . -k 3 -f 3 \n');
    return
end

% function returns estimated subgroups by hydra for clustering
% configurations ranging from K=1 to K=10, or another specified range of
% values. The function returns also the Adjusted Rand Index that was
% calculated across the cross-validation experiments and comparing
% respective clustering solutions.
%
% INPUT
%
% REQUIRED
% [--input, -i] : .csv file containing the input features. (REQUIRED)
%              every column of the file contains values for a feature, with
%              the exception of the first and last columns. We assume that
%              the first column contains subject identifying information
%              while the last column contains label information. First line
%              of the file should contain header information. Label
%              convention: -1 -> control group - 1 -> pathological group
%              that will be partioned to subgroups
% [--outputDir, -o] : directory where the output from all folds will be saved (REQUIRED)
%
% OPTIONAL
%
% [--covCSV, -z] : .csv file containing values for different covariates, which
%           will be used to correct the data accordingly (OPTIONAL). Every
%           column of the file contains values for a covariate, with the
%           exception of the first column, which contains subject
%           identifying information. Correction is performed by solving a
%           solving a least square problem to estimate the respective
%           coefficients and then removing their effect from the data. The
%           effect of ALL provided covariates is removed. If no file is
%           specified, no correction is performed.
%
% NOTE: featureCSV and covCSV files are assumed to have the subjects given
%       in the same order in their rows
%
% [--c, -c] : regularization parameter (positive scalar). smaller values produce
%     sparser models (OPTIONAL - Default 0.25)
% [--reg_type, -r] : determines regularization type. 1 -> promotes sparsity in the
%            estimated hyperplanes - 2 -> L2 norm (OPTIONAL - Default 1)
% [--balance, -b] : takes into account differences in the number between the two
%           classes. 1-> in case there is mismatch between the number of
%           controls and patient - 0-> otherwise (OPTIONAL - Default 1)
% [--init, -g] : initialization strategy. 0 : assignment by random hyperplanes
%        (not supported for regression), 1 : pure random assignment, 2:
%        k-means assignment, 3: assignment by DPP random
%        hyperplanes (default)
% [--iter, -t] : number of iterations between estimating hyperplanes, and cluster
%        estimation. Default is 50. Increase if algorithms fails to
%        converge
% [--numconsensus, -n] : number of clustering consensus steps. Default is 20.
%                Increase if algorithm gives unstable clustering results.
% [--kmin, -m] : determines the range of clustering solutions to evaluate
%             (i.e., kmin to kmax). Default  value is 1.
% [--kmax, -k] : determines the range of clustering solutions to evaluate
%             (i.e., kmin to kmax). Default  value is 10.
% [--kstep, -s] : determines the range of clustering solutions to evaluate
%             (i.e., kmin to kmax, with step kstep). Default  value is 1.
% [--cvfold, -f]: number of folds for cross validation. Default value is 10.
% [--vo, -j] : verbose output (i.e., also saves input data to verify that all were
%      read correctly. Default value is 0
% [--usage, -u]  Prints basic usage message.          
% [--help, -h]  Prints help information.
% [--version, -v]  Prints information about software version.
%
% OUTPUT:
% CIDX: sub-clustering assignments of the disease population (positive
%       class).
% ARI: adjusted rand index measuring the overlap/reproducibility of
%      clustering solutions across folds
%
% NOTE: to compile this function do
% mcc -m  hydra.m
%
%
% EXAMPLE USE (in matlab)
% hydra('-i','test.csv','-o','.','-k',3,'-f',3);
% EXAMPLE USE (in command line)
% hydra -i test.csv -o . -k 3 -f 3


params.kernel=0;



if( sum(or(strcmpi(varargin,'--input'),strcmpi(varargin,'-i')))==1)
    featureCSV=varargin{find(or(strcmpi(varargin,'--input'),strcmp(varargin,'-i')))+1};
else
    error('hydra:argChk','Please specify input csv file!');
end


if( sum(or(strcmpi(varargin,'--outputDir'),strcmpi(varargin,'-o')))==1)
    outputDir=varargin{find(or(strcmp(varargin,'--outputDir'),strcmp(varargin,'-o')))+1};
else
    error('hydra:argChk','Please specify output directory!');
end


if( sum(or(strcmpi(varargin,'--cov'),strcmpi(varargin,'-z')))==1)
    covCSV=varargin{find(or(strcmpi(varargin,'--cov'),strcmp(varargin,'-z')))+1};
else
    covCSV=[];
end

if( sum(or(strcmpi(varargin,'--c'),strcmpi(varargin,'-c')))==1)
    params.C=varargin{find(or(strcmpi(varargin,'--c'),strcmp(varargin,'-c')))+1};
else
    params.C=0.25;
end

if( sum(or(strcmpi(varargin,'--reg_type'),strcmpi(varargin,'-r')))==1)
    params.reg_type=varargin{find(or(strcmpi(varargin,'--reg_type'),strcmp(varargin,'-r')))+1};
else
    params.reg_type=1;
end

if( sum(or(strcmpi(varargin,'--balance'),strcmpi(varargin,'-b')))==1)
    params.balanceclasses=varargin{find(or(strcmpi(varargin,'--balance'),strcmp(varargin,'-b')))+1};
else
    params.balanceclasses=1;
end

if( sum(or(strcmpi(varargin,'--init'),strcmpi(varargin,'-g')))==1)
    params.init_type=varargin{find(or(strcmpi(varargin,'--init'),strcmp(varargin,'-g')))+1};
else
    params.init_type=3;
end

if( sum(or(strcmpi(varargin,'--iter'),strcmpi(varargin,'-t')))==1)
    params.numiter=varargin{find(or(strcmpi(varargin,'--iter'),strcmp(varargin,'-t')))+1};
else
    params.numiter=50;
end

if( sum(or(strcmpi(varargin,'--numconsensus'),strcmpi(varargin,'-n')))==1)
    params.numconsensus=varargin{find(or(strcmpi(varargin,'--numconsensus'),strcmp(varargin,'-n')))+1};
else
    params.numconsensus=20;
end

if( sum(or(strcmpi(varargin,'--kmin'),strcmpi(varargin,'-m')))==1)
    params.kmin=varargin{find(or(strcmpi(varargin,'--kmin'),strcmp(varargin,'-m')))+1};
else
    params.kmin=1;
end

if( sum(or(strcmpi(varargin,'--kstep'),strcmpi(varargin,'-s')))==1)
    params.kstep=varargin{find(or(strcmpi(varargin,'--kstep'),strcmp(varargin,'-s')))+1};
else
    params.kstep=1;
end

if( sum(or(strcmpi(varargin,'--kmax'),strcmpi(varargin,'-k')))==1)
    params.kmax=varargin{find(or(strcmpi(varargin,'--kmax'),strcmp(varargin,'-k')))+1};
else
    params.kmax=10;
end

if( sum(or(strcmpi(varargin,'--cvfold'),strcmpi(varargin,'-f')))==1)
    params.cvfold=varargin{find(or(strcmpi(varargin,'--cvfold'),strcmp(varargin,'-f')))+1};
else
    params.cvfold=10;
end

if( sum(or(strcmpi(varargin,'--vo'),strcmpi(varargin,'-j')))==1)
    params.vo=varargin{find(or(strcmpi(varargin,'--vo'),strcmp(varargin,'-j')))+1};
else
    params.vo=0;
end

% create output directory
if (~exist(outputDir,'dir'))
    [status,~,~] = mkdir(outputDir);
    if (status == 0)
        error('hydra:argChk','Cannot create output directory!');
    end
end


params.C=input2num(params.C);
params.reg_type=input2num(params.reg_type);
params.balanceclasses=input2num(params.balanceclasses);
params.init_type=input2num(params.init_type);
params.numiter=input2num(params.numiter);
params.numconsensus=input2num(params.numconsensus);
params.kmin=input2num(params.kmin);
params.kstep=input2num(params.kstep);
params.kmax=input2num(params.kmax);
params.cvfold=input2num(params.cvfold);
params.vo=input2num(params.vo);


% confirm validity of optional input arguments
validateFcn_reg_type = @(x) (x==1) || (x == 2);
validateFcn_balance = @(x) (x==0) || (x == 1);
validateFcn_init = @(x) (x==0) || (x == 1) || (x==2) || (x == 3) || (x == 4);
validateFcn_iter = @(x) isscalar(x) && (x>0) && (mod(x,1)==0);
validateFcn_consensus = @(x) isscalar(x) && (x>0) && (mod(x,1)==0);
validateFcn_kmin = @(x) isscalar(x) && (x>0) && (mod(x,1)==0);
validateFcn_kmax = @(x,y) isscalar(x) && (x>0) && (mod(x,1)==0) && (x>y);
validateFcn_kstep = @(x,y,z) isscalar(x) && (x>0) && (mod(x,1)==0) && (x+y<z);
validateFcn_cvfold = @(x) isscalar(x) && (x>0) && (mod(x,1)==0);
validateFcn_vo = @(x) (x==0) || (x == 1);

if(~validateFcn_reg_type(params.reg_type))
    error('hydra:argChk','Input regularization type (reg_type) should be either 1 or 2!');
end
if(~validateFcn_balance(params.balanceclasses))
    error('hydra:argChk','Input balance classes (balance) should be either 1 or 2!');
end
if(~validateFcn_init(params.init_type))
    error('hydra:argChk','Initialization type can be either 0, 1, 2, 3, or 4!');
end
if(~validateFcn_iter(params.numiter))
    error('hydra:argChk','Number of iterations should be a positive integer!');
end
if(~validateFcn_consensus(params.numconsensus))
    error('hydra:argChk','Number of clustering consensus steps should be a positive integer!');
end
if(~validateFcn_kmin(params.kmin))
    error('hydra:argChk','Minimum number of clustering solutions to consider should be a positive integer!');
end
if(~validateFcn_kmax(params.kmax,params.kmin))
    error('hydra:argChk','Maximum number of clustering solutions to consider should be a positive integer that is greater than the minimum number of clustering solutions!');
end
if(~validateFcn_kstep(params.kstep,params.kmin,params.kmax))
    error('hydra:argChk','Step number of clustering solutions to consider should be a positive integer that is between the minimun and maximum number of clustering solutions!');
end
if(~validateFcn_cvfold(params.cvfold))
    error('hydra:argChk','Number of folds for cross-validation should be a positive integer!');
end
if(~validateFcn_vo(params.vo))
    error('hydra:argChk','VO parameter should be either 0 or 1!');
end

disp('Done');
disp('HYDRA runs with the following parameteres');
disp(['featureCSV: ' featureCSV]);
disp(['OutputDir: ' outputDir]);
disp(['covCSV: ' covCSV])
disp(['C: ' num2str(params.C)]);
disp(['reg_type: ' num2str(params.reg_type)]);
disp(['balanceclasses: ' num2str(params.balanceclasses)]);
disp(['init_type: ' num2str(params.init_type)]);
disp(['numiter: ' num2str(params.numiter)]);
disp(['numconsensus: ' num2str(params.numconsensus)]);
disp(['kmin: ' num2str(params.kmin)]);
disp(['kmax: ' num2str(params.kmax)]);
disp(['kstep: ' num2str(params.kstep)]);
disp(['cvfold: ' num2str(params.cvfold)]);
disp(['vo: ' num2str(params.vo)]);

% csv with features
fname=featureCSV;
if (~exist(fname,'file'))
    error('hydra:argChk','Input feature .csv file does not exist');
end

% csv with features
covfname=covCSV;
if(~isempty(covfname))
    if(~exist(covfname,'file'))
        error('hydra:argChk','Input covariate .csv file does not exist');
    end
end

% input data
% assumption is that the first column contains IDs, and the last contains
% labels
disp('Loading features...');
input=readtable(fname);
ID=input{:,1};
XK=input{:,2:end-1};
Y=input{:,end};

% z-score imaging features
XK=zscore(XK);
disp('Done');

% input covariate information if necesary
if(~isempty(covfname))
    disp('Loading covariates...');
    covardata = readtable(covfname) ;
    IDcovar = covardata{:,1};
    covar = covardata{:,2:end};
    covar = zscore(covar);
    disp('Done');
end

% NOTE: we assume that the imaging data and the covariate data are given in
% the same order. No test is performed to check that. By choosing to have a
% verbose output, you can have access to the ID values are read by the
% software for both the imaging data and the covariates

% verify that we have covariate data and imaging data for the same number
% of subjects
if(~isempty(covfname))
    if(size(covar,1)~=size(XK,1))
        error('hydra:argChk','The feature .csv and covariate .csv file contain data for different number of subjects');
    end
end

% residualize covariates if necessary
if(~isempty(covfname))
    disp('Residualize data...');
    [XK0,~]=GLMcorrection(XK,Y,covar,XK,covar);
    disp('Done');
else
    XK0=XK;
end

% for each realization of cross-validation
clustering=params.kmin:params.kstep:params.kmax;
part=make_xval_partition(size(XK0,1),params.cvfold); %Partition data to 10 groups for cross validation
% for each fold of the k-fold cross-validation
disp('Run HYDRA...');
for f=1:params.cvfold
    % for each clustering solution
    for kh=1:length(clustering)
        params.k=clustering(kh);
        disp(['Applying HYDRA for ' num2str(params.k) ' clusters. Fold: ' num2str(f) '/' num2str(params.cvfold)]);
        model=hydra_solver(XK0(part~=f,:),Y(part~=f,:),[],params);
        YK{kh}(part~=f,f)=model.Yhat;
    end
end
disp('Done');

disp('Estimating clustering stabilitiy...')
% estimate cluster stability for the cross-validation experiment
ARI = zeros(length(clustering),1);
for kh=1:length(clustering)
    tmp=cv_cluster_stability(YK{kh}(Y~=-1,:));
    ARI(kh)=tmp(1);
end
disp('Done')

disp('Estimating final consensus group memberships...')
% Computing final consensus group memberships
CIDX=-ones(size(Y,1),length(clustering)); %variable that stores subjects in rows, and cluster memberships for the different clustering solutions in columns
for kh=1:length(clustering)
    CIDX(Y==1,kh)=consensus_clustering(YK{kh}(Y==1,:),clustering(kh));
end
disp('Done')

disp('Saving results...')
if(params.vo==0)
    save([outputDir '/HYDRA_results.mat'],'ARI','CIDX','clustering','ID');
else
    save([outputDir '/HYDRA_results.mat'],'ARI','CIDX','clustering','ID','XK','Y','covar','IDcovar');
end
disp('Done')
end

function [score,stdscore]=cv_cluster_stability(S)
k=0;
for i=1:size(S,2)-1
    for j=i+1:size(S,2)
        k=k+1;
        zero_idx=any([S(:,i) S(:,j)]==0,2);
        [a(k),b(k),c(k),d(k)]=RandIndex(S(~zero_idx,i),S(~zero_idx,j));
    end
end
score=[mean(a) mean(b) mean(c) mean(d)];
stdscore=[std(a) std(b) std(c) std(d)];
end

function [AR,RI,MI,HI]=RandIndex(c1,c2)
%RANDINDEX - calculates Rand Indices to compare two partitions
% ARI=RANDINDEX(c1,c2), where c1,c2 are vectors listing the
% class membership, returns the "Hubert & Arabie adjusted Rand index".
% [AR,RI,MI,HI]=RANDINDEX(c1,c2) returns the adjusted Rand index,
% the unadjusted Rand index, "Mirkin's" index and "Hubert's" index.
%
% See L. Hubert and P. Arabie (1985) "Comparing Partitions" Journal of
% Classification 2:193-218

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

if nargin < 2 | min(size(c1)) > 1 | min(size(c2)) > 1
    error('RandIndex: Requires two vector arguments')
    return
end

C=Contingency(c1,c2);	%form contingency matrix

n=sum(sum(C));
nis=sum(sum(C,2).^2);		%sum of squares of sums of rows
njs=sum(sum(C,1).^2);		%sum of squares of sums of columns

t1=nchoosek(n,2);		%total number of pairs of entities
t2=sum(sum(C.^2));	%sum over rows & columnns of nij^2
t3=.5*(nis+njs);

%Expected index (for adjustment)
nc=(n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1));

A=t1+t2-t3;		%no. agreements
D=  -t2+t3;		%no. disagreements

if t1==nc
    AR=0;			%avoid division by zero; if k=1, define Rand = 0
else
    AR=(A-nc)/(t1-nc);		%adjusted Rand - Hubert & Arabie 1985
end

RI=A/t1;			%Rand 1971		%Probability of agreement
MI=D/t1;			%Mirkin 1970	%p(disagreement)
HI=(A-D)/t1;	%Hubert 1977	%p(agree)-p(disagree)

    function Cont=Contingency(Mem1,Mem2)
        
        if nargin < 2 | min(size(Mem1)) > 1 | min(size(Mem2)) > 1
            error('Contingency: Requires two vector arguments')
            return
        end
        
        Cont=zeros(max(Mem1),max(Mem2));
        
        for i = 1:length(Mem1);
            Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
        end
    end
end

function IDXfinal=consensus_clustering(IDX,k)
[n,~]=size(IDX);
cooc=zeros(n);
for i=1:n-1
    for j=i+1:n
        cooc(i,j)=sum(IDX(i,:)==IDX(j,:));
    end
    %cooc(i,i)=sum(IDX(i,:)==IDX(i,:))/2;
end
cooc=cooc+cooc';
L=diag(sum(cooc,2))-cooc;

Ln=eye(n)-diag(sum(cooc,2).^(-1/2))*cooc*diag(sum(cooc,2).^(-1/2));
Ln(isnan(Ln))=0;
[V,~]=eig(Ln);
try
    IDXfinal=kmeans(V(:,1:k),k,'emptyaction','drop','replicates',20);
catch
    disp('Complex Eigenvectors Found...Using Non-Normalized Laplacian');
    [V,~]=eig(L);
    IDXfinal=kmeans(V(:,1:k),k,'emptyaction','drop','replicates',20);
end

end

function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

s=mod(n,n_folds);r=n-s;
p1=ceil((1:r)/ceil(r/n_folds));
p2=randperm(n_folds);p2=p2(1:s);
p3=[p1 p2];
part=p3(randperm(size(p3,2)));
end

function [X0train,X0test]=GLMcorrection(Xtrain,Ytrain,covartrain,Xtest,covartest)

X1=Xtrain(Ytrain==-1,:);
C1=covartrain(Ytrain==-1,:);
B=[C1 ones(size(C1,1),1)];
Z=X1'*B*inv(B'*B);
X0train=(Xtrain'-Z(:,1:end-1)*covartrain')';
X0test=(Xtest'-Z(:,1:end-1)*covartest')';
end

function printhelp()
fprintf(' function returns estimated subgroups by hydra for clustering \n')
fprintf(' configurations ranging from K=1 to K=10, or another specified range of\n')
fprintf(' values. The function returns also the Adjusted Rand Index that was\n')
fprintf(' calculated across the cross-validation experiments and comparing\n')
fprintf(' respective clustering solutions.\n')
fprintf('\n')
fprintf(' INPUT\n')
fprintf('\n')
fprintf(' REQUIRED\n')
fprintf(' [--input, -i] : .csv file containing the input features. (REQUIRED)\n')
fprintf('              every column of the file contains values for a feature, with\n')
fprintf('              the exception of the first and last columns. We assume that\n')
fprintf('              the first column contains subject identifying information\n')
fprintf('              while the last column contains label information. First line\n')
fprintf('              of the file should contain header information. Label\n')
fprintf('              convention: -1 -> control group - 1 -> pathological group\n')
fprintf('              that will be partioned to subgroups\n')
fprintf(' [--outputDir, -o] : directory where the output from all folds will be saved (REQUIRED)\n')
fprintf('\n')
fprintf(' OPTIONAL\n')
fprintf('\n')
fprintf(' [--covCSV, -z] : .csv file containing values for different covariates, which\n')
fprintf('           will be used to correct the data accordingly (OPTIONAL). Every\n')
fprintf('           column of the file contains values for a covariate, with the\n')
fprintf('           exception of the first column, which contains subject\n')
fprintf('           identifying information. Correction is performed by solving a\n')
fprintf('           solving a least square problem to estimate the respective\n')
fprintf('           coefficients and then removing their effect from the data. The\n')
fprintf('           effect of ALL provided covariates is removed. If no file is\n')
fprintf('           specified, no correction is performed.\n')
fprintf('\n')
fprintf(' NOTE: featureCSV and covCSV files are assumed to have the subjects given\n')
fprintf('       in the same order in their rows\n')
fprintf('\n')
fprintf(' [--c, -c] : regularization parameter (positive scalar). smaller values produce\n')
fprintf('     sparser models (OPTIONAL - Default 0.25)\n')
fprintf(' [--reg_type, -r] : determines regularization type. 1 -> promotes sparsity in the\n')
fprintf('            estimated hyperplanes - 2 -> L2 norm (OPTIONAL - Default 1)\n')
fprintf(' [--balance, -b] : takes into account differences in the number between the two\n')
fprintf('           classes. 1-> in case there is mismatch between the number of\n')
fprintf('           controls and patient - 0-> otherwise (OPTIONAL - Default 1)\n')
fprintf(' [--init, -g] : initialization strategy. 0 : assignment by random hyperplanes\n')
fprintf('        (not supported for regression), 1 : pure random assignment, 2:\n')
fprintf('        k-means assignment, 3: assignment by DPP random\n')
fprintf('        hyperplanes (default)\n')
fprintf(' [--iter, -t] : number of iterations between estimating hyperplanes, and cluster\n')
fprintf('        estimation. Default is 50. Increase if algorithms fails to\n')
fprintf('        converge\n')
fprintf(' [--numconsensus, -n] : number of clustering consensus steps. Default is 20.\n')
fprintf('                Increase if algorithm gives unstable clustering results.\n')
fprintf(' [--kmin, -m] : determines the range of clustering solutions to evaluate\n')
fprintf('             (i.e., kmin to kmax). Default  value is 1.\n')
fprintf(' [--kmax, -k] : determines the range of clustering solutions to evaluate\n')
fprintf('             (i.e., kmin to kmax). Default  value is 10.\n')
fprintf(' [--kstep, -s] : determines the range of clustering solutions to evaluate\n')
fprintf('             (i.e., kmin to kmax, with step kstep). Default  value is 1.\n')
fprintf(' [--cvfold, -f]: number of folds for cross validation. Default value is 10.\n')
fprintf(' [--vo, -j] : verbose output (i.e., also saves input data to verify that all were\n')
fprintf('      read correctly. Default value is 0\n')
fprintf(' [--usage, -u]  Prints basic usage message.     \n');     
fprintf(' [--help, -h]  Prints help information.\n');
fprintf(' [--version, -v]  Prints information about software version.\n');
fprintf('\n')
fprintf(' OUTPUT:\n')
fprintf(' CIDX: sub-clustering assignments of the disease population (positive\n')
fprintf('       class).\n')
fprintf(' ARI: adjusted rand index measuring the overlap/reproducibility of\n')
fprintf('      clustering solutions across folds\n')
fprintf('\n')
fprintf(' NOTE: to compile this function do\n')
fprintf(' mcc -m  hydra.m\n')
fprintf('\n')
fprintf('\n')
fprintf(' EXAMPLE USE (in matlab)\n')
fprintf(' hydra(''-i'',''test.csv'',''-o'',''.'',''-k'',3,''-f'',3);\n')
fprintf(' EXAMPLE USE (in command line)\n')
fprintf(' hydra -i test.csv -o . -k 3 -f 3\n')
fprintf('======================================================================\n');
fprintf('Contact: software@cbica.upenn.edu\n');
fprintf('\n');
fprintf('Copyright (c) 2018 University of Pennsylvania. All rights reserved.\n');
fprintf('See COPYING file or http://www.med.upenn.edu/sbia/software/license.html\n');
fprintf('======================================================================\n');
end

function o=input2num(x)
if isnumeric(x)
    o=x;
else
    o = str2double(x);
end
end
