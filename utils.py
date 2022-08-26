import numpy as np
import matplotlib.patches as patches
import tensorflow as tf

def PDFF_at_ROI(X,left_x,sup_y):
  r1,r2 = sup_y,(sup_y+9)
  c1,c2 = left_x,(left_x+9)
  PDFF_crop = X[r1:r2,c1:c2]
  return np.median(PDFF_crop)

def R2_at_ROI(X,left_x,sup_y):
  r1,r2 = sup_y,(sup_y+9)
  c1,c2 = left_x,(left_x+9)
  R2_crop = X[r1:r2,c1:c2]
  return np.mean(R2_crop)

class IndexTracker(object):
  def __init__(self, fig, ax, X, PDFF_bool, lims, npy_file='slices_crops.npy'):
    self.fig = fig
    self.ax = ax
    ax.set_title('use scroll wheel to navigate images')

    self.X = X
    rows, cols, self.slices = X.shape
    self.ind = 0 # self.slices//2

    try:
      with open(npy_file,'rb') as f:
        frms = np.load(f)
        crops_1 = np.load(f)
        crops_2 = np.load(f)
      self.frms = list(frms)
      self.crops_1 = list(crops_1)
      self.crops_2 = list(crops_2)
    except FileNotFoundError:
      print('No previously existent crops.')
      self.frms = []
      self.crops_1 = []
      self.crops_2 = []
    
    self.flag = False
    self.fflag = False

    self.flag2 = False
    self.fflag2 = False

    self.saveFlag = False
    self.eraseFlag = False

    self.PDFF_bool = PDFF_bool
    vmin, vmax = lims
    self.im = ax.imshow(self.X[:, :, self.ind],vmin=vmin,vmax=vmax)
    self.fig.colorbar(self.im,ax=self.ax)
    self.update()

  def onscroll(self, event):
    # print("%s %s" % (event.button, event.step))
    if event.button == 'up':
      self.ind = (self.ind + 1) % self.slices
    else:
      self.ind = (self.ind - 1) % self.slices
    self.ax.patches = []
    self.fflag = False
    self.fflag2 = False
    self.update()

  def button_press(self, event):
    # print("%s %s" % (event.button, event.step))
    if event.button == 1:
      r_ct = np.round(event.xdata)
      c_ct = np.round(event.ydata)
      self.left_x1 = int(r_ct - 4)
      self.sup_y1 = int(c_ct - 4)
      self.rect_gt_1 = patches.Rectangle((self.left_x1,self.sup_y1),9,9,
        linewidth=1.5,edgecolor='r',facecolor='none')
      self.flag = True
      self.update()
    elif event.button == 3:
      r_ct = np.round(event.xdata)
      c_ct = np.round(event.ydata)
      self.left_x2 = int(r_ct - 4)
      self.sup_y2 = int(c_ct - 4)
      self.rect_gt_2 = patches.Rectangle((self.left_x2,self.sup_y2),9,9,
        linewidth=1.5,edgecolor='orange',facecolor='none')
      self.flag2 = True
      self.update()

  def key_press(self,event):
    if (event.key == 'up') or (event.key == 'down'):
      if event.key == 'up':
        self.ind = (self.ind - 1) % self.slices
      elif event.key == 'down':
        self.ind = (self.ind + 1) % self.slices
      self.ax.patches = []
      self.fflag = False
      self.fflag2 = False
      self.update()
    elif event.key == 'v':
      if self.ind in self.frms:
        idx = self.frms.index(self.ind)
        self.frms.pop(idx)
        self.crops_1.pop(idx)
        self.crops_2.pop(idx)
      self.frms.append(self.ind)
      self.crops_1.append([self.left_x1,self.sup_y1])
      self.crops_2.append([self.left_x2,self.sup_y2])
      self.saveFlag = True
      self.update()
    # - - - - 
    elif event.key == 'b':
      if self.ind in self.frms:
        idx = self.frms.index(self.ind)
        self.frms.pop(idx)
        self.crops_1.pop(idx)
        self.crops_2.pop(idx)
        self.ax.patches = []
        self.fflag = False
        self.fflag2 = False
        self.eraseFlag = True
        self.update()
    # - - - - 

  def update(self):
    self.im.set_data(self.X[:, :, self.ind])
    # - - - - 
    if (self.ind in self.frms) and (not(self.flag)) and (not(self.flag2)):
      idx = self.frms.index(self.ind)
      self.flag = True
      self.flag2 = True
      self.left_x1 = self.crops_1[idx][0]
      self.sup_y1 = self.crops_1[idx][1]
      self.left_x2 = self.crops_2[idx][0]
      self.sup_y2 = self.crops_2[idx][1]
      self.rect_gt_1 = patches.Rectangle((self.left_x1,self.sup_y1),9,9,
        linewidth=1.5,edgecolor='r',facecolor='none')
      self.rect_gt_2 = patches.Rectangle((self.left_x2,self.sup_y2),9,9,
        linewidth=1.5,edgecolor='orange',facecolor='none')
    # - - - -
    if self.flag:
      # Calculate PDFF at ROI
      if self.PDFF_bool:
        self.PDFF = PDFF_at_ROI(self.X[:,:,self.ind],self.left_x1,self.sup_y1)
      else:
        self.PDFF = R2_at_ROI(self.X[:,:,self.ind],self.left_x1,self.sup_y1)
      if self.fflag:
        self.ax.patches = self.ax.patches[-1:]
      self.ax.add_patch(self.rect_gt_1)
      self.ax.patches = self.ax.patches[::-1]
      self.flag = False
      self.fflag = True
    if self.flag2:
      # Calculate PDFF at ROI
      if self.PDFF_bool:
        self.PDFF2 = PDFF_at_ROI(self.X[:,:,self.ind],self.left_x2,self.sup_y2)
      else:
        self.PDFF2 = R2_at_ROI(self.X[:,:,self.ind],self.left_x2,self.sup_y2)
      if self.fflag2:
        self.ax.patches = self.ax.patches[:-1]
      self.ax.add_patch(self.rect_gt_2)
      self.flag2 = False
      self.fflag2 = True
    if self.saveFlag:
      self.ax.set_title('Crops successfully saved!')
      self.saveFlag = False
    elif self.eraseFlag:
      self.ax.set_title('Crops of these frame were successfully deleted')
      self.eraseFlag = False
    elif self.fflag and self.fflag2:
      if self.PDFF_bool:
        self.ax.set_title('PDFF_1 = '+str(np.round(self.PDFF*100,2))+'% - '+
          ' PDFF_2 = '+str(np.round(self.PDFF2*100,2))+'%')
      else:
        self.ax.set_title('R2*_1 = '+str(np.round(self.PDFF,2))+'[1/s] - '+
          ' R2*_2 = '+str(np.round(self.PDFF2,2))+'[1/s]')
    else:
      self.ax.set_title('use scroll wheel to navigate images')
    self.ax.set_ylabel('slice %s' % self.ind)
    self.im.axes.figure.canvas.draw()


class IndexTracker_phantom(object):
  def __init__(self, fig, ax, X, PDFF_bool, lims, npy_file='slices_crops.npy'):
    self.fig = fig
    self.ax = ax
    ax.set_title('use scroll wheel to navigate images')

    self.X = X
    rows, cols, self.slices = X.shape
    self.ind = 0 # self.slices//2

    try:
      with open(npy_file,'rb') as f:
        frms = np.load(f)
        crops_1 = np.load(f)
        crops_2 = np.load(f)
      self.frms = list(frms)
      self.crops_1 = list(crops_1)
      self.crops_2 = list(crops_2)
    except FileNotFoundError:
      print('No previously existent crops.')
      self.frms = []
      self.crops_1 = []
      self.crops_2 = []
    
    self.flag = False
    self.fflag = False # Si ya había una ROI clickeada, pero no guardada (mem inmediata)

    self.saveFlag = False
    self.eraseFlag = False

    self.recentSave = False

    self.PDFF_bool = PDFF_bool
    vmin, vmax = lims
    self.im = ax.imshow(self.X[:, :, self.ind],vmin=vmin,vmax=vmax)
    self.fig.colorbar(self.im,ax=self.ax)
    self.update()

  def onscroll(self, event):
    # print("%s %s" % (event.button, event.step))
    if event.button == 'up':
      self.ind = (self.ind + 1) % self.slices
    else:
      self.ind = (self.ind - 1) % self.slices
    self.ax.patches = []
    self.fflag = False
    self.fflag2 = False
    self.update()

  def button_press(self, event):
    # print("%s %s" % (event.button, event.step))
    if event.button == 1:
      r_ct = np.round(event.xdata)
      c_ct = np.round(event.ydata)
      self.left_x1 = int(r_ct - 4)
      self.sup_y1 = int(c_ct - 4)
      self.rect_gt_1 = patches.Rectangle((self.left_x1,self.sup_y1),9,9,
        linewidth=1.5,edgecolor='r',facecolor='none')
      self.flag = True
      # self.fflag = True
      self.update()

  def key_press(self,event):
    if (event.key == 'up') or (event.key == 'down'):
      if event.key == 'up':
        self.ind = (self.ind - 1) % self.slices
      elif event.key == 'down':
        self.ind = (self.ind + 1) % self.slices
      self.ax.patches = []
      self.fflag = False
      self.update()
    elif event.key == 'v':
      self.frms.append(self.ind)
      self.crops_1.append([self.left_x1,self.sup_y1])
      self.rect_gt_1 = patches.Rectangle((self.left_x1,self.sup_y1),9,9,
        linewidth=1.5,edgecolor='orange',facecolor='none')
      self.saveFlag = True
      self.update()
    # - - - - 
    elif event.key == 'b':
      idxs = [i for i,x in enumerate(self.frms) if x==self.ind]
      for i in idxs[::-1]:
        self.frms.pop(i)
        self.crops_1.pop(i)
      self.ax.patches = []
      self.fflag = False
      self.eraseFlag = True
      self.update()
    # - - - - 

  def update(self):
    self.im.set_data(self.X[:, :, self.ind])
    # - - - - 
    if (self.ind in self.frms) and not(self.flag):
      idxs = [i for i,x in enumerate(self.frms) if x==self.ind]
      for idx in idxs:
        self.ax.add_patch(patches.Rectangle((self.crops_1[idx][0],self.crops_1[idx][1]),9,9,
          linewidth=1.5,edgecolor='orange',facecolor='none'))

    # - - - -
    if self.flag:
      # Calculate PDFF at ROI
      if self.PDFF_bool:
        self.PDFF = PDFF_at_ROI(self.X[:,:,self.ind],self.left_x1,self.sup_y1)
      else:
        self.PDFF = R2_at_ROI(self.X[:,:,self.ind],self.left_x1,self.sup_y1)
      if self.fflag and not(self.recentSave):
        self.ax.patches = self.ax.patches[:-1]
      self.ax.add_patch(self.rect_gt_1)
      self.flag = False
      self.fflag = True
      self.recentSave = False
    
    if self.saveFlag:
      self.ax.set_title('Crop successfully saved: PDFF='+str(np.round(self.PDFF*100,2))+'%')
      self.saveFlag = False
      self.ax.patches = self.ax.patches[:-1]
      self.ax.add_patch(self.rect_gt_1)
      self.recentSave = True
    elif self.eraseFlag:
      self.ax.set_title('Crops of these frame were successfully deleted')
      self.eraseFlag = False
    else:
      self.ax.set_title('use scroll wheel to navigate images')
    self.ax.set_ylabel('slice %s' % self.ind)
    self.im.axes.figure.canvas.draw()
