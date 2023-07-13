rm(list=ls(all=TRUE))

library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

model = "/Sup-012/"
map = "PDFF"
epoch = "100"

dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

ls_sheets = excel_sheets(paste(map,'_selPhantom_ROIs.xlsx',sep=""))
for (i in c(1:length(ls_sheets)))
{
	roi_data=read_excel(paste(map,'_phantom_ROIs.xlsx',sep=""),sheet=ls_sheets[i])
	if (i==1)
	{
		if (map=="PDFF") {refs = c(t(roi_data[,1]))}
		else {refs = c(t(roi_data[,2]))}
		meas = c(t(roi_data[,3]))
		im_id = rep(c(i),length(t(roi_data[,1])))
		vial_id = c(t(1:length(t(roi_data[,1]))))
	} else
	{
		# meas = meas + c(t(roi_data[,3]))
		if (map=="PDFF") {refs = c(refs,t(roi_data[,1]))}
		else {refs = c(refs,t(roi_data[,2]))}
		meas = c(meas,t(roi_data[,3]))
		im_id = append(im_id,rep(c(i),length(t(roi_data[,1]))))
		vial_id = c(vial_id,t(1:length(t(roi_data[,1]))))
	}
}
# meas = meas/length(ls_sheets)
n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
mean = c(meas),
refs = c(refs),
bias = c(meas-refs),
vial = factor(c(vial_id)),
Site_Prot = factor(c(im_id),labels=c("GE","Philips","Siemens"))
)

# Dataset summary stats
pdff_Vials <- pdff_Data %>%
  group_by(vial) %>%
  summarise(meanY = mean(mean),
		refsX = mean(refs),
		bias = mean(mean)-mean(refs))
overall_bias = mean(pdff_Vials$bias)
std_bias = sd(pdff_Vials$bias)
cat('Overall bias:',overall_bias,'+-',std_bias,'\n')


############################################################################
########################### REGRESSION LINES ###############################
############################################################################
if (map=="PDFF") {yl=1.0} else {yl=40.0}
q = ggplot(pdff_Data, aes(refs, mean, colour=Site_Prot)) +
       geom_point()+
       geom_smooth(method="lm", se=FALSE, fullrange=TRUE) +
	 theme(text = element_text(size = 12)) +
	 ylim(0.0,yl) +
	 ylab("Measurements") +
	 xlab("References") +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"))
    )
fn1 = paste(map,"selPhan-corr.png",sep="-")
ggsave(plot=q, width=5, height=3, dpi=400, filename=fn1)

