
dca <- function(data, f){
    
baseline.model <- decision_curve(f,
                                 data = data, 
                                 bootstraps = 50,
                                 confidence.intervals=NA)

#plot the curve
plot_decision_curve(baseline.model,  curve.names = "baseline model")

}