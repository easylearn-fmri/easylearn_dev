function el_glm()
% General linear model for 3D data

% Inputs
if nargin < 1
    cov_all = 'D:\workstation_b\duanjia\test\test.xlsx';
    patient = 'D:\workstation_b\duanjia\test\2patient';
    hc = 'D:\workstation_b\duanjia\test\1hc';
    mask = 'D:\workstation_b\duanjia\BrainMask_05_91x109x91.img';
    out_path = 'D:\workstation_b\duanjia';
end

%% load
cov_all = importdata(cov_all);

pd = dir(patient);
pn = {pd.name}';
pp = fullfile(patient, pn);
pp = pp(3:end);
ppd = y_ReadAll(pp);

hcd = dir(hc);
hcn = {hcd.name}';
hcp = fullfile(hc, hcn);
hcp = hcp(3:end);
hcpd = y_ReadAll(hcp);

[mask3d,header]  = y_Read(mask);
mask3d = mask3d > 0;

% Filter
ppd = reshape(ppd, [], size(ppd,4))';
hcpd = reshape(hcpd, [], size(hcpd,4))';
mask = reshape(mask3d, 1, []);

ppd = ppd(:,mask);
hcpd = hcpd(:, mask);

% NBS parameters
y = cat(1, hcpd, ppd);
x = cov_all.data(:,2:end);
x = cat(2,x, x(:,1).*x(:,4));

independent_variables = [cat(1,ones(size(hcpd,1),1),zeros(size(ppd,1),1)),cat(1,zeros(10,1),ones(10,1)), x(:,2:end)];
dependent_variables=y;
test_type = 'ttest'; 
contrast = [0 0 0 0 0 1];

[teststat, pvalues] = el_glm(independent_variables, dependent_variables, contrast, test_type);

[Results] = multcomp_fdr_bh(pvalues);
H = Results.corrected_h;

% Save
H3d = zeros(size(mask3d));
P3d = ones(size(mask3d));
tstat3d = zeros(size(mask3d));
H3d(mask3d) = H;
P3d(mask3d) = pvalues;
tstat3d(mask3d) = teststat;
header.dt=[16,0];
[n,m]=size(x);
df = n-m-1;
header.descrip=sprintf('DPABI{T_[%d]}',df);
y_Write(H3d, header, fullfile(out_path, 'H.nii'));
y_Write(P3d, header, fullfile(out_path, 'P.nii'));
y_Write(tstat3d, header, fullfile(out_path, 'tstat.nii'));
