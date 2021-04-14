function DECISIONCURVE(x, label, varargin)
% 

if nargin==3
    color = varargin{1};
    showlegend = false;
else
    color = 'r';
    showlegend = true;
end
dd = xlsread('./demodata.xlsx');
x = dd(:,1);
label = dd(:,2);

DecisionCurve = @(TP, FP, N, Pt)TP/N - FP/N * Pt/(1 - Pt);
% DecisionCurve = inline('TP/N - FP/N * Pt/(1 - Pt)', 'TP', 'FP', 'N', 'Pt');
% x = 0:0.1:1;
% label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
N = length(x);
numberOfPt = 100;
step = (max(x) - min(x)) / numberOfPt;
Pt = 0;
NetBenefit = [];
PT = [];
for i=1:numberOfPt-1
    predict = x>=Pt;
    TP = numel(find(label==1&predict==1));
    FP = numel(find(label==0&predict==1));
    PT(:, end+1) = i / numberOfPt;
    NetBenefit(:, end+1) = DecisionCurve(TP, FP, N, PT(i));
    Pt = Pt + step;
end
%%
plot(PT, NetBenefit, color, 'LineWidth', 2);hold on;
%% Treat None
plot(PT, linspace(0, 0, numberOfPt-1), 'k', 'LineWidth', 2);
%% Treat All
PT = [];
NetBenefit = [];
TP = numel(find(label==1));
FP = numel(find(label==0));
for i=1:numberOfPt-1
    PT(:, end+1) = i / numberOfPt;
    NetBenefit(:, end+1) = DecisionCurve(TP, FP, N, PT(i));
end
plot(PT, NetBenefit, 'b', 'LineWidth', 2);hold off;
%%
axis([0, 1, -0.05, max(NetBenefit)+0.1]);
xlabel('Threshold Probability', 'FontSize',16);
ylabel('Net Benefit', 'FontSize',16);
if showlegend
    legend('Radiomics', 'None', 'All');
end

