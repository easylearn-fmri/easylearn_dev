% Hosmer-Lemeshow goodness-of-fit test
% M. Farbood, April 29, 2012
% 
% Assumes that the last column in M is the outcome.  The previous columns
% are the predictors.  
%
function [HLstat, HLp, contingencyM] = HLtest1(M, numGroups)

    % Assumes there are 10 groups by default if a value is not specified.
    if ~exist('numGroups')
        numGroups = 10;
    end

    % Note this balances out the size of the group as much as possible.  
    % Change the values in numObservationsPerGroup vector if you want to hard code
    % boundaries.
    [N, cols] = size(M);
    n = floor(N/numGroups);
    leftovers = N - (n*numGroups);
    noExtrasBoundary = numGroups-leftovers;
    numObservationsPerGroup = ones(numGroups,1) * n;
    if noExtrasBoundary ~= numGroups
        numObservationsPerGroup(noExtrasBoundary+1:end) = numObservationsPerGroup(noExtrasBoundary+1:end)+1;
    end

    % This is an approximation of the groups from the article Peng et al.,
    % 2002, "An Introduction to Logistic Regression
    % Analysis and Reporting" in the Journal of Educational Research.
    % They managed to have 90% of the groups have expected frequencies of 5 or
    % higher.  It was unclear how this was possible given the groups sizes they
    % provided and keeping data sorted by expected frequencies.  
    % I did my best approximation. Uncomment this line to use this grouping for
    % the special case for Peng et al., 2002.
%     numObservationsPerGroup = [21 20 20 20 20 20 19 19 19 11];

    % Sort by expected values
    [sortedM, iSort] = sort(M(:, 1));
    sortedObs = M(iSort,cols);
    sortedM = [sortedM sortedObs];

    % Caluculate contingency table for the groups
    currIndex = 1;
    contingencyM = [];
    for j=1:numGroups
        numObservations = numObservationsPerGroup(j);
        g = sortedM(currIndex:currIndex+numObservations-1,:);

        predict = sum(g(:,1));
        observed = sum(g(:,2));

        % Add values for group to the contingency table
        contingencyM = [contingencyM; j numObservations predict observed];

        currIndex = currIndex + numObservations;
    end

    % Hosmer-Lemeshow statistic = sum over all groups of (O-E)^2/(E(1-E/n))
    HLstat = sum((contingencyM(:,4) - contingencyM(:,3)).^2./(contingencyM(:,3).*(1-contingencyM(:,3)./contingencyM(:,2))));% chi-square value
    HLdf = numGroups - 2; % degree of freedom
    HLp = 1 - chi2cdf(HLstat,HLdf);

    %fprintf('Hosmer-Lemeshow chisq = %g  p = %g  df = %d  N = %d\n', HLchisq, HLp, df, N);

