rm(list=ls(all=TRUE))

library(readxl)
library(lme4)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

models = c("/Sup-200/","/Sup-202/","/Sup-204/")
map = "PDFF"
epoch = "100"

for (k in c(1:length(models)))
{
dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",models[k],"Ep-",epoch,sep="")
setwd(dir)

ls_sheets = excel_sheets(paste(map,'_phantom_ROIs.xlsx',sep=""))
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
		meth_id = rep(c(k),length(t(roi_data[,1])))
	} else
	{
		# meas = meas + c(t(roi_data[,3]))
		if (map=="PDFF") {refs = c(refs,t(roi_data[,1]))}
		else {refs = c(refs,t(roi_data[,2]))}
		meas = c(meas,t(roi_data[,3]))
		im_id = append(im_id,rep(c(i),length(t(roi_data[,1]))))
		vial_id = c(vial_id,t(1:length(t(roi_data[,1]))))
		meth_id = append(meth_id,rep(c(k),length(t(roi_data[,1]))))
	}
}
}
# meas = meas/length(ls_sheets)
n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
bias = c(meas-refs),
vial = factor(c(vial_id)),
method = factor(c(meth_id)),
Site_Prot = factor(c(im_id),labels=c("S1-P1","S1-P2",
		   "S2-P1","S2-P2","S3-P1","S3-P2","S4-P2","S6-P1","S6-P2"))
)
# factor(c(im_id),labels=c("S1-P1(V1)","S1-P2(V1)","S1-P1(V2)","S1-P2(V2)",
# 		   "S2-P1","S2-P2","S3-P1","S3-P2","S4-P2","S6-P1","S6-P2"))

# Dataset summary stats
pdff_Vials <- pdff_Data %>%
  group_by(vial) %>%
  summarise(bias = mean(mean)-mean(refs))
overall_bias = mean(pdff_Vials$bias)
std_bias = sd(pdff_Vials$bias)
cat('Overall bias:',overall_bias,'+-',std_bias,'\n')
