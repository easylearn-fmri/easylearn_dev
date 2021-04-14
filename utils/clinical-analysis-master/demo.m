clear;clc;
load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4) + rand(100,1)*3;
y = species(inds);
Y(strcmp(y, 'versicolor')) = 1;
Y(strcmp(y, 'virginica')) = 0;
mdl = fitglm(X,Y','Distribution','binomial');
probs = predict(mdl, X);
%% Calibration curve
[HLstat, HLp, contingencyM] = HLtest1([probs, Y'], 5); % set degree of freedom
% The larger HLp(P value), the better the calibration of the model.
x = contingencyM(:, 3);
y = contingencyM(:, 4);
%% curve fit
% y1 = polyfit(x, y, 2);
% y = polyval(y1, x);
%%
plot(x/ceil(max(x)), y/ceil(max(y)), 'LineWidth', 2);hold on;
plot([0, 1], [0, 1], '--', 'LineWidth', 2);
axis([0 1 0 1]);
title('Calibration Curve', 'FontSize',16);
xlabel('Radiomics-Predicted Probability', 'FontSize',16);
ylabel('Actual Rate of Grade 2', 'FontSize',16);hold off;
%% Decision Curve
figure;DECISIONCURVE(probs, Y');
%% Visualization
x = linspace(-40, 40, 1000);
y = 1./(1 + exp(-x));
figure;plot(x, y);ylim([-0.1, 1.1]);hold on;
plot(x, 0.5*ones(1, length(x)), '-.');
[B, I] = sort(probs);
sortedLabel = Y(I)';
for i=1:numel(probs)
    if sortedLabel(i),color='r';r=i;mkr='o';else,color='b';g=i;mkr='+';end
    temp(i) = scatter(-log(1/B(i)-1), B(i), color, mkr);
end
legend([temp(r), temp(g)],{'1', '2'}, 'Location', 'best');
hold off;