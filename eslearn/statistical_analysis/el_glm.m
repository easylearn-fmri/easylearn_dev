function [teststat, pvalues, beta_value] = el_glm(independent_variables, dependent_variables, contrast, test_type)
% General linear model for fmri-like (Very high dimension of dependent variables, e.g. voxels) data
% This function is based on revised NBSglm, so users should cite NBS software.
% USAGE: el_glm(dependent_variables, independent_variables, contrast, test_type)
% INPUT:
%   independent_variables: matrix of independent variables with n-by-m dimension, n is observations, m is number of variables
%   dependent_variables: matrix of dependent variables with n-by-p dimension, n is observations, p is number of variables
%   contrast: vector of contrast with 1-by-m dimension
%   test_type: string of test type 'onesample' or 'ttest' or 'ftest'. Please refer to NBS software.
% OUTPUT:
%   teststat: T-statistic or F-statistic
%   pvalues: p-values of corresponding test
% 

GLM.test = test_type; 
GLM.contrast = contrast;
GLM.X = independent_variables;
GLM.y=dependent_variables;
[teststat, pvalues, beta_value]=NBSglm_x(GLM);
end

function [test_stat,pvalues, beta_value]=NBSglm_x(varargin)
%NBSglm_x is revised from NBSglm
%
%   Test_Stat=NBSglm_x(GLM) operates on each GLM defined by the structure
%   GLM.
%
%   A GLM structure contains the following fields:
%       GLM.X:            n x p design matrix, including a column of ones
%                         if necessary. p is the number of independent
%                         variables, n is the number of observations.
%       GLM.y:            n x M array, where each column stores the
%                         dependent variables for a seperate GLM
%       GLM.contrast:     1 x p contrast vector, only one contrast can be
%                         specified
%       GLM.test:         Type of test: {'onesample','ttest','ftest'}


GLM=varargin{1};

%Number of predictors (including intercept)
p=length(GLM.contrast);

%Number of independent GLM's to fit
M=size(GLM.y,2);

%Number of observations
n=size(GLM.y,1);

%Determine nuisance predictors not in contrast
ind_nuisance=find(~GLM.contrast);

if isempty(ind_nuisance)
    %No nuisance predictors
else
    %Regress out nuisance predictors and compute residual
    b=zeros(length(ind_nuisance),M);
    resid_y=zeros(n,M);
    b=GLM.X(:,ind_nuisance)\GLM.y;
    resid_y=GLM.y-GLM.X(:,ind_nuisance)*b;
end

test_stat=zeros(1,M);

dependent_variables=GLM.y;
    
if ~isempty(ind_nuisance)
    dependent_variables=resid_y+[GLM.X(:,ind_nuisance)]*b;
end

beta_value=zeros(p,M);
beta_value=GLM.X\dependent_variables;

%Compute statistic of interest
if strcmp(GLM.test,'onesample')
    test_stat=mean(dependent_variables);

elseif strcmp(GLM.test,'ttest')
    resid=zeros(n,M);
    mse=zeros(n,M);
    resid=dependent_variables-GLM.X*beta_value;
    mse=sum(resid.^2)/(n-p);
    se=sqrt(mse*(GLM.contrast*inv(GLM.X'*GLM.X)*GLM.contrast'));
    test_stat=(GLM.contrast*beta_value)./se;  % bj/Sbj, df=n-m-1
    % Added by Li Chao
    if nargout >= 2
        pvalues = 2*(1-tcdf(abs(test_stat),n-p-1));  % df=n-m-1; The reason why I multiplied it by 2 is that I applied two-tailed test.
    end

elseif strcmp(GLM.test,'ftest')
    sse=zeros(1,M);
    ssr=zeros(1,M);
    %Sum of squares due to error
    sse=sum((dependent_variables-GLM.X*beta_value).^2);
    %Sum of square due to regression
    ssr=sum((GLM.X*beta_value-repmat(mean(dependent_variables),n,1)).^2);
    if isempty(ind_nuisance)
        test_stat=(ssr/(p-1))./(sse/(n-p));
        % Added by Li Chao
        if nargout >= 2
            pvalues = 1-fcdf(test_stat,p-1,n-p);
        end
    else
        %Get reduced model
        %Column of ones will be added to the reduced model unless the
        %resulting matrix is rank deficient
        X_new=[ones(n,1),GLM.X(:,ind_nuisance)];
        %+1 because a column of 1's will be added to the reduced model
        b_red=zeros(length(ind_nuisance)+1,M);
        %Number of remaining variables
        v=length(find(GLM.contrast))-1;
        [n,ncolx]=size(X_new);
        [Q,R,perm]=qr(X_new,0);
        rankx = sum(abs(diag(R)) > abs(R(1))*max(n,ncolx)*eps(class(R)));
        if rankx < ncolx
            %Rank deficient, remove column of ones
            X_new=GLM.X(:,ind_nuisance);
            b_red=zeros(length(ind_nuisance),M);
            v=length(find(GLM.contrast));
        end
        
        sse_red=zeros(1,M);
        ssr_red=zeros(1,M);
        b_red=X_new\dependent_variables;
        sse_red=sum((dependent_variables-X_new*b_red).^2);
        ssr_red=sum((X_new*b_red-repmat(mean(dependent_variables),n,1)).^2);
        test_stat=((ssr-ssr_red)/v)./(sse/(n-p));
        % added by Li Chao
        if nargout >= 2
            pvalues =1-fcdf(test_stat,v,n-p);
        end                  
    end
end

%Added to v1.1.2
%Covers the case where the dependent variable is identically zero for all
%observations. The test statistic in this case in NaN. Therefore, force any
%NaN elements to zero. This case is typical of connectivity matrices
%populated using streamline counts, in which case some regional pairs are
%not interconnected by any streamlines for all subjects.
test_stat(isnan(test_stat))=0;
end