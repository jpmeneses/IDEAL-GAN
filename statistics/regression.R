library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(emmeans)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

model = "/TEaug-001/"
epoch = "75"

dir = paste("C:/Users/jpmen/Documents/OT-CycleGAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

rhl_data=read_excel('PDFF_ROIs.xlsx',sheet='RHL')
lhl_data=read_excel('PDFF_ROIs.xlsx',sheet='LHL')

refs <- c(t(rhl_data[,1]),t(lhl_data[,1]))*100 # 3ech: R2s -> del c(-9,-22)
meas <- c(t(rhl_data[,2]),t(lhl_data[,2]))*100
n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
mean = c(meas),
refs = c(refs)
)

# Dataset summary stats
pdff_Data %>%
  get_summary_stats(mean, type="common")
 
############################################################################
########################### REGRESSION LINES ###############################
############################################################################
q = ggscatter(
  pdff_Data, x="refs", y="mean",
  color="darkorange1",
  xlab="Reference PDFF [%]",
  ylab="Measured PDFF [%]",
  add="reg.line",
  add.params=list(color="blue"),
  conf.int=FALSE,
  )+
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~")),
    color="blue"
    )
fn1 = "LS-corr.png"
ggsave(plot=q, width=3, height=3, dpi=300, filename=fn1)

############################################################################
######################### BLAND ALTMAN PLOTS ###############################
############################################################################
# Averages and differences
pdff_Data$avg <- (pdff_Data$mean + pdff_Data$refs)/2
pdff_Data$df <- (pdff_Data$mean - pdff_Data$refs)

# create Bland-Altman plot
mean_diff = mean(pdff_Data$df)
lower <- mean_diff - 1.96*sd(pdff_Data$df)
upper <- mean_diff + 1.96*sd(pdff_Data$df)
r=ggplot(pdff_Data, aes(x = avg, y = df)) +
  geom_point(size=1) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  ylab("Bias PDFF [%]") +
  xlab("Mean PDFF [%]") +
  xlim(0,45) +
  ylim(-5,5)
fn2 = "BlandAltman-R.png"
ggsave(plot=r, width=3, height=3, dpi=300, filename=fn2)
