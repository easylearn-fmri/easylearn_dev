function lc_NBSstat(x, y, cov, perms, contrast, test_type, STATS)
% Modified from NBS
% Refer and thanks to NBS (NBSglm and NBSstats)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% debug
if nargin < 1
    x = [[ones(50,1);zeros(50,1)], [zeros(50,1);ones(50,1)]];
    y1=[rand(50,441)+10,rand(50,6000)];
    y2=[rand(50,441)+100,rand(50,6000)];
    y = cat(1,y1,y2);
    cov = rand(100,3);
    perms = 100;
    contrast = [1 -1  0 0 0 ];
    test_type = 'ttest';
    % NBS stat parameters
    STATS.thresh = 3;
    STATS.alpha = 0.05;
    STATS.N = 114;
    STATS.size = 'extent';
end

% GLM parameters
GLM.X = cat(2, x, cov);
GLM.y = y;
GLM.perms = perms;
GLM.contrast = contrast;
GLM.test = test_type;

% GLM 
[test_stat,P]=NBSglm(GLM);
sig_loc = P <= 0.05;
test_stat(sig_loc);

% NBS to test_stat (or pval)
STATS.test_stat = test_stat;
[n_cnt,con_mat,pval]=NBSstats(STATS);
end