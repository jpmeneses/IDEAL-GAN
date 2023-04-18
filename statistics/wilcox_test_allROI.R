rm(list=ls(all=TRUE))

library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(gridExtra)

model = "/Sup-007/"
map = "PDFF"
epoch = "200"

dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

TEs_suffix = c('13_21','13_22','13_23','13_24','14_21','14_22')
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

# Check extreme outliers
pdff_Data %>%
  group_by(TEs) %>%
  identify_outliers(refs)

# Check normality assumption (Shapiro-Wilk test) per protocol
pdff_Data %>%
  group_by(TEs) %>%
  shapiro_test(meas)