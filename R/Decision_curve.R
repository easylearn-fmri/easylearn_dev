library(rmda)

#load simulated data 
data(dcaData)

head(dcaData)

set.seed(123)
#first use rmda with the default settings (set bootstraps = 50 here to reduce computation time). 
baseline.model <- decision_curve(Cancer~Age + Female + Smokes, #fitting a logistic model
                                 data = dcaData, 
                                 study.design = "cohort", 
                                 policy = "opt-in",  #default 
                                 bootstraps = 50,
                                 confidence.intervals=NA)

#plot the curve
plot_decision_curve(baseline.model,  curve.names = "baseline model")


set.seed(123)
full.model <- decision_curve(Cancer~Age + Female + Smokes + Marker1 + Marker2,
                             data = dcaData, 
                             bootstraps = 50,
                             confidence.intervals=NA)

#since we want to plot more than one curve, we pass a list of 'decision_curve' objects to the plot
plot_decision_curve( list(baseline.model, full.model), 
                     curve.names = c("Baseline model", "Full model"), xlim = c(0, 1), 
                     legend.position = "bottomright") 