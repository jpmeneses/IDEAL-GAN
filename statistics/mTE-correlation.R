rm(list=ls(all=TRUE))

library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(gridExtra)
#library(emmeans)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

model = "/Sup-007/"
map = "PDFF"
epoch = "200"

dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

TEs_suffix = c('13_21','13_22','14_22')
for (i in c(1:length(TEs_suffix)))
{
  filename = paste(c(map,'_ROIs_',TEs_suffix[i],'.xlsx'),collapse="")
  ROI_sheets = excel_sheets(filename)
  for (j in c(1:length(ROI_sheets)))
  {
    roi_data=read_excel(filename,sheet=ROI_sheets[j])
    len_ij = length(t(roi_data[,2]))
    if (i==1 & j==1)
    {
      refs = c(t(roi_data[,1]))
	meas = c(t(roi_data[,2]))
	sample_id = c(1:len_ij)
      TEs = rep(c(TEs_suffix[i]),len_ij)
    } else
    {
      refs = c(refs,t(roi_data[,1]))
      meas = c(meas,t(roi_data[,2]))
	sample_id = c(sample_id,c(1:len_ij)+(j-1)*len_ij)
      TEs = c(TEs,rep(c(TEs_suffix[i]),len_ij))
    }
  }
}
n_data = length(meas)

# Create a data frame
pdff_Data <- data.frame(
refs = c(refs),
meas = c(meas),
sample_id = factor(sample_id,
			 labels=c(paste0(LETTERS[1:len_ij],sep='-',1),
				    paste0(LETTERS[1:len_ij],sep='-',2))
			 ),
id = factor(sample_id, labels=rep(LETTERS[1:len_ij],2)),
roi = factor(c(rep(c("RHL"),len_ij),rep(c("LHL"),len_ij))),
TEs = factor(TEs)
)

# Remove zero-values
pdff_Data = subset(pdff_Data, meas > 0.0)

# Dataset summary stats
pdff_Data %>%
  group_by(sample_id) %>%
  get_summary_stats(meas, type="common")

############################################################################
############################## SCATTERPLOT #################################
############################################################################
q = ggplot(pdff_Data, aes(refs, meas, colour=TEs)) +
      geom_point()+
	geom_smooth(method="lm",fullrange=TRUE) +
	labs(x='Reference', y='Measurement') +
	theme(text = element_text(size = 12)) +
	xlim(0.0,0.3) +
	ylim(0.0,0.3) +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"))
    )
ggsave(plot=q, width=6, height=4, dpi=300, filename="mTE-corr.png")

