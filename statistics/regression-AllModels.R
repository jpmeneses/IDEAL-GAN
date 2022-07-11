library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(emmeans)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

#setwd("C:/Users/jpmen/Documents/PDFF-models/Radiology models/3ech_tests")
setwd("C:/Users/jpmen/Desktop/JuanPablo/UC/PhD/ISMRM/2022/Abstract")

#gfg_data=read_excel('All_R2s_ROIs.xlsx',sheet="Mean")
gfg_data=read_excel('PDFF_ROIs.xlsx',sheet="RHL")

refs <- c(t(gfg_data[,1]))*100 # 3ech: R2s -> del c(-9,-22)
otcy <- c(t(gfg_data[,2]))*100
mdwf <- c(t(gfg_data[,4]))*100
unet <- c(t(gfg_data[,3]))*100

n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
Method = factor(rep(c(0,1,2), each=n_data), labels=c("OT-CycleGAN","MDWF-Net","U-Net")),
mean = c(otcy,mdwf,unet),
refs = rep(c(refs),3)
)

# Dataset summary stats
pdff_Data %>%
  group_by(Method) %>%
  get_summary_stats(mean, type="common")
 
############################################################################
########################### REGRESSION LINES ###############################
############################################################################
q = ggscatter(
  pdff_Data, x="refs", y="mean",
  color="Method", xlab="Reference PDFF [%]",
  ylab="Measured R2* [%]", add="reg.line",
  )+
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"), color=Method)
    )
ggsave(plot=q, width=4, height=4, dpi=300, filename="LS-corr.png")

############################################################################
######################### BLAND ALTMAN PLOTS ###############################
############################################################################
# Averages and differences
pdff_Data$avg <- (pdff_Data$mean + pdff_Data$refs)/2
pdff_Data$df <- (pdff_Data$mean - pdff_Data$refs)

# create OT-CycleGAN Bland-Altman
otcy_Data <- pdff_Data %>% filter(Method == "OT-CycleGAN")
mean_diff = mean(otcy_Data$df)
lower <- mean_diff - 1.96*sd(otcy_Data$df)
upper <- mean_diff + 1.96*sd(otcy_Data$df)
r=ggplot(otcy_Data, aes(x = avg, y = df)) +
  geom_point(size=1) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  ylab("Bias PDFF [%]") +
  xlab("Mean PDFF [%]") +
  xlim(0,0.45) +
  ylim(-0.05,0.05)
r
ggsave(plot=r, width=2, height=2, dpi=300, filename="OTcGAN-BlandAltman.png")

# create MDWF-Net Bland-Altman
mdwf_Data <- pdff_Data %>% filter(Method == "MDWF-Net")
mean_diff = mean(mdwf_Data$df)
lower <- mean_diff - 1.96*sd(mdwf_Data$df)
upper <- mean_diff + 1.96*sd(mdwf_Data$df)
r=ggplot(mdwf_Data, aes(x = avg, y = df)) +
  geom_point(size=1) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  ylab("Bias PDFF [%]") +
  xlab("Mean PDFF [%]") +
  xlim(0,0.45) +
  ylim(-0.05,0.05)
r
ggsave(plot=r, width=2, height=2, dpi=300, filename="MDWF-BlandAltman.png")

# create U-Net Bland-Altman
unet_Data <- pdff_Data %>% filter(Method == "U-Net")
mean_diff = mean(unet_Data$df)
lower <- mean_diff - 1.96*sd(unet_Data$df)
upper <- mean_diff + 1.96*sd(unet_Data$df)
s=ggplot(unet_Data, aes(x = avg, y = df)) +
  geom_point(size=1) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  ylab("Bias PDFF [%]") +
  xlab("Mean PDFF [%]") +
  xlim(0,0.45) +
  ylim(-0.05,0.05)
s
ggsave(plot=s, width=2, height=2, dpi=300, filename="UNet-BlandAltman.png")