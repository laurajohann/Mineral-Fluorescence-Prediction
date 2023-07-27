#!/usr/bin/env python
# coding: utf-8
#%%
# # Library Builder for Mineral Fluorescense Prediction
# ### Builder app that allows you to select image folder containing training images. Images can be filtered and enhanced in tkinter interface before pixel values are measured and stored. At the end of image file, app allows you to save the library in a folder which will automatically save library plot and mineral list.
# 
# #### Select specific folder with images to optain training data from. Images can be manipulated (saturation control with slide bar, deleting specific pixel areas with cursur, and blurring to exclude bad pixels). With deleting pixels, double click for single selection, click and drag for large area. Before adding imiage values to dataset, user can check HSV for the displayed image by selecting 'check'. Image pixel data can be added to library with 'add'. App displayd the progression of library as it is built. 

 #%% 
import PIL
import tkinter
import cv2
from tkinter import *
from PIL import Image, ImageGrab, ImageDraw
from PIL import ImageTk
from tkinter import filedialog
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from scipy import stats as st
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from skimage import img_as_ubyte, img_as_uint, img_as_float64, transform
from skimage.transform import rescale, resize
from matplotlib.figure import Figure
from sklearn import preprocessing
from skimage.color import (rgb2hsv,rgb2lab,rgb2rgbcie)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from skimage import io  
import customtkinter
from CTkRangeSlider.ctk_rangeslider import CTkRangeSlider
from CTkRangeSlider import *
import os
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import colorcet as cc
pwd = os.getcwd()
#%% 

### Functions

#%% 
def select_file(tp):
    global df, hfltr,vfltr, blur, folder, counter, pixel_x, pixel_y, x, y,w,h
    if tp == 'lib':
        df =  pd.read_csv(filedialog.askopenfilename(initialdir= f'{pwd}\\Libraries\\'))
        folder = filedialog.askdirectory(initialdir= f'{pwd}\\Catalog\\')
        Label(tags_frame,text="Task: Adding to Existing Library").grid(row=3, column=0)
        print(df['Mineral'].unique())
        disp_lib()
        
    if tp == 'new':
        df = pd.DataFrame(columns=['r', 'g', 'b', 'short', 'med', 'lon','L', 'A', 'B',
                        'hue', 'saturation', 'value','Mineral'])
        folder = filedialog.askdirectory(initialdir= f'{pwd}\\Catalog\\')
        Label(tags_frame,text="Task: Creating New Library").grid(row=3, column=0)
    w=im_frame.winfo_width()
    h=im_frame.winfo_height()
    file_imgs()    
    
def file_imgs():
    global folder, list_images, counter, max_count, hfltr,vfltr, blur, pixel_x, pixel_y, x, y
    list_images = []
    for file in os.listdir(folder):
         if file.endswith('png') or file.endswith('jpg'):
            list_images.append(file)
    max_count = len(list_images)-1
    disp_img()
    
def disp_img():
    global folder, list_images, im, w, h,counter, hfltr,vfltr, blur, canvas, pixel_x, pixel_y, x, y,img_name
    im_path = f'{folder}\\{list_images[counter]}'
    im = io.imread(im_path)
    img_name =Label(tags_frame,text='Image Name:' + str(list_images[counter]) )
    img_name.grid(row=3, column = 1)
    canvas = Canvas(im_frame, width=w, height=h, borderwidth=0)
    canvas.grid(row=0, column=0)
    update_image()
        
def update_image():
    global im, w, h, hfltr,vfltr, blur, pixel_x, pixel_y, x, y, adapted_im, im_filtered, canvas
    im_filtered = filter_pixel(im,w,h,hfltr,vfltr,blur,pixel_x, pixel_y) 
    adapted_im = ImageTk.PhotoImage(Image.fromarray(img_as_ubyte(im_filtered)))
    canvas.config(width=adapted_im.width(), height=adapted_im.height())
    canvas.create_image(0, 0, image=adapted_im, anchor=NW)

def hue_filter(hvar):
    global im, hfltr,vfltr, blur, pixel_x, pixel_y, x, y
    
    #hfltr = [100-hvar[1], hvar[1]] 
    #hue_values = Label(tool_frame, text = f"Hue Quantile Filter: {min(hfltr): .2f} - {max(hfltr): .2f}")
    #hue_values.grid(row=3, column=1, columnspan=2)
    #hue_slider.set(hfltr)
    #update_image()
    
def v_filter(vvar):
    global im, hfltr,vfltr, blur, pixel_x, pixel_y, x, y
    vfltr = vvar
    v_values = Label(tool_frame, text = f"Value Filter: {min(vfltr): .2f} - {max(vfltr): .2f}")
    v_values.grid(row=1, column=1, columnspan=2)
    update_image()

def change_blur():
    global im, hfltr,vfltr, blur, pixel_x, pixel_y, x, y
    blur = blur_slider.get()
    update_image()

def select_pixels():  
    global im, hfltr,vfltr, blur, adapted_im,pixel_x, pixel_y, x, y, canvas
    canvas.bind("<B1-Motion>", save_cord)
    canvas.bind("<Double-Button-1>", save_cord)
    #canvas.bind("<Control-z>", lambda e: del_pixel())

    
def del_pixel():
    global im, hfltr,vfltr, blur, pixel_x, pixel_y
    mid = len(pixel_x)/2
    pixel_x= pixel_x[mid:-1]
    pixel_y = pixel_y[mid:-1]
    update_image()
    
def save_cord(event):
    global im, hfltr,vfltr, blur, pixel_x, pixel_y, x, y
    x = int(event.x)
    y = int(event.y)
    pixel_x.append(x)
    pixel_y.append(y)
    #pixel_frame = Frame(tool_frame, width=2, height=5)
    #pixel_frame.grid(row=3, column=1)
    #x_list = Listbox(pixel_frame,listvariable=StringVar(value=pixel_x))
    #x_list.pack(side="left")
    #y_list = Listbox(pixel_frame,listvariable=StringVar(value=pixel_y))
    #y_list.pack(side="left")
    update_image()

def add_img():
    global df, df_current, im_filtered,pixel_x, pixel_y, x, y, mineral_root, mineral_var,mineral_name,fig
    mineral_root = Toplevel(root)
    mineral_root.title("Name")
    mineral_frame = Frame(mineral_root, width=300, height=500)
    mineral_frame.pack(side = TOP, expand=True, fill="both")
    Label(mineral_frame,text="Mineral Name:").grid(row=0, column=0)
    mineral_var = StringVar()
    mineral_entry = Entry(mineral_frame,textvariable=mineral_var)
    mineral_entry.grid(row=0, column=1)
    for widget in lib_frame.winfo_children():
            widget.destroy()
    mineral_root.bind('<Return>',lambda e: store_rgb())
    
def store_rgb():
    global df, df_current, im_filtered,pixel_x, pixel_y, x, y, mineral_root, mineral_var,mineral_name
    mineral_name = mineral_var.get()
    print(mineral_name)
    mineral_root.destroy()
    
    df_current = pd.DataFrame()
    read_rgb()
    df_current['Mineral'] = mineral_name
    
    #test_value = df.columns.to_list()
    #test_value.remove('Mineral') 
    #df_current[test_value] = df_current[test_value].round(5)
    
    df_add = df_current.drop_duplicates()
    df = pd.concat([df,df_add], ignore_index = True)
    disp_lib()
    
def disp_lib():
    global df,fig
    
    plt.close(fig)
    
    for widget in lib_frame.winfo_children():
            widget.destroy()
            
    width= lib_frame.winfo_width()
    height= lib_frame.winfo_height()
    
    palette = sns.color_palette(cc.glasbey, n_colors=len(df['Mineral'].unique()))
    sns.set_theme(style = "ticks", palette=palette)
    sns.set_context("notebook", font_scale = 0.65)
    
    test_value = df.columns.to_list()
    test_value.remove('Mineral')
    n=len(test_value)
    rows= 3 - 1*(n==4)
    cols = int(np.ceil(n/rows))
    fig, ax = plt.subplots(figsize=(width/100,height/100), dpi=100,nrows=rows, ncols=cols,constrained_layout = True)
    v=0
    for i in range(0,rows):
        for j in range(0,cols):
            x = str(test_value[int(v)])
            fig.suptitle('Mineral Library')
            sns.histplot(data=df, x=x, hue = 'Mineral', kde=True, fill=False, stat="density", ax=ax[i,j],
                     element="step",legend=False,hue_order= df['Mineral'].unique())
            ax[i,j].set_xlabel(x)
            v = v+1
    f = FigureCanvasTkAgg(fig, master = lib_frame)
    f.get_tk_widget().grid(sticky=NS)
    f.draw()
    
def disp_current():
    global df_current, mineral_name
    
    for widget in vals_frame.winfo_children():
            widget.destroy()

    read_rgb()
    width= vals_frame.winfo_width()
    height= vals_frame.winfo_height()
    fig0, ax0 =  plt.subplots(figsize=(width/100,height/100), dpi=100)
    
    alphas =  np.round_(df_current['saturation'].sort_values(), decimals = 1).unique()
    for i in range(0,len(alphas)):
        dt=df_current[(df_current['saturation'] <= alphas[i]) & (df_current['saturation'] > alphas[i-1])]
        x,y,z = dt['hue'].values,dt['value'].values, dt['hue'].values
        points = ax0.scatter(x*360, y, c=z*360, s=5, cmap="gist_rainbow",alpha=alphas[i],vmin=0,vmax=360)
    
    #fig0.colorbar(points)
    ax0.set_title('Current Sample HSV')
    ax0.set(ylabel='Value')
    ax0.set(xlabel='Hue')
    ax0.set_xlim(0,360)
    f0 = FigureCanvasTkAgg(fig0, master = vals_frame)
    f0.get_tk_widget().grid(sticky=NS)
    f0.draw()
    plt.close(fig0)
    
def next_img():
    global list_images, im, counter, counter_max, hfltr,vfltr, blur, pixel_x, pixel_y, x, y
    counter += 1
    if counter > max_count:
        max_root = Toplevel(root)
        max_root.title("End of File")
        max_frame = Frame(max_root, width=300, height=300)
        max_frame.pack(side = TOP, expand=True, fill="both")
        Label(max_frame,text="You Reached End of Folder").grid(row=0, column=0, padx=5, pady=5)
        Button(max_frame, text="Save this Library",command=save_lib).grid(row=1, column=2, padx=5, pady=5)
    else:
        #hfltr = [0,100]
        vfltr = [0,100]
        blur = 0
        x = IntVar
        y = IntVar
        pixel_x = []
        pixel_y = []
        
        #hue_slider.set(hfltr)
        #hue_values=Label(tool_frame, text = f"Hue Quantile Filter:{min(hfltr): .2f} - {max(hfltr): .2f}")
        #hue_values.grid(row=3, column=1, columnspan=2)
        
        v_slider.set(vfltr)
        v_values=Label(tool_frame, text = f"Value Filter:{min(vfltr): .2f} - {max(vfltr): .2f}")
        v_values.grid(row=1, column=1, columnspan=2)
        
        blur_slider.set(blur)
        for widget in vals_frame.winfo_children():
            widget.destroy()
        disp_img()


def undo_img():
    global im, hfltr,vfltr, blur, pixel_x, pixel_y, x, y, adapted_im, im_filtered, canvas, counter
    counter -= 1
    #hfltr = [0,100]
    vfltr = [0,100]
    blur=0
    pixel_x = []
    pixel_y = []
    x = IntVar
    y = IntVar
    
    #hue_values.config(text = f"Hue Quantile Filter: {min(hfltr): .2f} - {max(hfltr): .2f}")
    #hue_values.grid(row=3, column=1, columnspan=2)
    #hue_slider.set(hfltr)

    v_values.config(text = f"Value Filter: {min(vfltr): .2f} - {max(vfltr): .2f}")
    v_values.grid(row=1, column=1, columnspan=2)
    v_slider.set(vfltr)
    
    blur_slider.set(blur)
    update_image()
    
def save_lib():
    global df, fig, save_root
    save_root = Toplevel(root)
    save_root.title("End of File")
    save_frame = Frame(save_root, width=300, height=300)
    save_frame.pack(side = TOP, expand=True, fill="both")
    Label(save_frame,text="Name Library Folder:").grid(row=0, column=0)
    save_folder = StringVar()
    save_entry = Entry(save_frame,textvariable = save_folder).grid(row=0, column=1)
    save_root.bind('<Return>',lambda e: saving(save_folder.get(),df,fig))

def saving(save_folder,df,fig):
    global save_root
    from pathlib import Path
    p = Path(f'{pwd}\\Libraries\\') / save_folder
    p.mkdir(exist_ok=True)
    save_dt = p / 'data_set.csv'
    save_plot = p / 'hue_plot.png'
    
    mineral_list = pd.DataFrame({'Minerals in Library': df['Mineral'].unique()})
    save_list = p / 'mineral_list.txt'
    
    df.to_csv(save_dt,index=False)
    fig.savefig(save_plot)
    mineral_list.to_csv(save_list,index=False)
    save_root.destroy()

def size(img,w,h):
    fixed_height = h
    fixed_width = w
    height_percent = (fixed_height / float(img.shape[1]))
    width_percent = (fixed_width / float(img.shape[0]))
    if height_percent > width_percent:
        width_size = int((float(img.shape[0]) * float(height_percent)))
        return resize(img, (width_size, fixed_height,img.shape[2] ), anti_aliasing=True)
    if width_percent > height_percent:
        height_size = int((float(img.shape[1]) * float(width_percent)))
        return resize(img, (fixed_width,height_size,img.shape[2] ), anti_aliasing=True)
    if width_percent == height_percent:
        return resize(img, (fixed_width, fixed_height, img.shape[2] ), anti_aliasing=True)
    
def filter_pixel(img,w,h,hfltr,vfltr,blur,x,y):
    image = size(img,w,h)
    
    if blur > 1:
        if (blur % 2) == 0:
            blur += 1
        ksize = np.full(2, blur, dtype=int)
        image = cv2.GaussianBlur(image,ksize,cv2.BORDER_DEFAULT)                 
    img = Image.fromarray((image* 255).astype(np.uint8))
    for i in range(len(x)):
        seed = (x[i],y[i])
        rep_value = (0, 0, 0)
        ImageDraw.floodfill(img, seed, rep_value, thresh = 50)
        
    im_64 = img_as_float64(img) 
    
    im_hsv = rgb2hsv(im_64)
    h = im_hsv[:,:,0]
    s = im_hsv[:,:,1]
    v = im_hsv[:,:,2]
    
    #hmx = max(hfltr)
    #hmn = min(hfltr)
    #q1, q3 = np.percentile(sorted(h.ravel()), [hmn, hmx])
    # compute IRQ
    #iqr = q3 - q1
    # find lower and upper bounds
    #hlower = q1 - (1.5 * iqr)
    #hupper = q3 + (1.5 * iqr)
    
    vmx = max(vfltr)
    vmn = min(vfltr)
    vlower = vmn/100
    vupper = vmx/100
    
    good =  (v > vlower)  & (v <= vupper) #& (h < hupper )  & (h >= hlower) 
    
    gd = np.zeros(im_64.shape, dtype = int)
    gd[:,:,0] = good
    gd[:,:,1] = good
    gd[:,:,2] = good
    
    filtered = im_64*gd   
    return filtered

def read_rgb():
    global df,df_current,im_filtered,mineral_root, mineral_name
    img = size(im_filtered,300,300)
    img = img_as_float64(img)
    array_img = np.asarray(img)
    hsv_img = rgb2hsv(img)
    lab_img = rgb2lab(img)
    rgbcie_img = rgb2rgbcie(img)
    rgb_img = img
    
    
    image = hsv_img
    pixels = np.asarray(image).max()
    h = image[:,:,0]/np.max(pixels)
    s = image[:,:,1]/np.max(pixels)
    v = image[:,:,2]/np.max(pixels)
    
    good = v>0.01
    
    hue = h[good].ravel()
    sat = s[good].ravel()
    val = v[good].ravel()

    image = lab_img
    pixels = np.asarray(np.abs(image)).max()
    l = image[:,:,0]/np.max(pixels)
    a = image[:,:,1]/np.max(pixels)
    b = image[:,:,2]/np.max(pixels)
    L = l[good].ravel()
    A = a[good].ravel()
    B = b[good].ravel()

    image = rgbcie_img
    pixels = np.asarray(image).max()
    s = image[:,:,0]/np.max(pixels)
    m = image[:,:,1]/np.max(pixels)
    l = image[:,:,2]/np.max(pixels)
    short = s[good].ravel()
    med = m[good].ravel()
    lon = l[good].ravel()

    image = rgb_img
    pixels = np.asarray(image).max()
    red = image[:,:,0]/np.max(pixels)
    green = image[:,:,1]/np.max(pixels)
    blue = image[:,:,2]/np.max(pixels)
    r = red[good].ravel()
    g = green[good].ravel()
    b = blue[good].ravel()


    df_current = pd.DataFrame({ 'r': r, 'g': g, 'b': b, 
                        'short': short, 'med': med, 'lon': lon,
                        'L': L, 'A': A, 'B': B,
                        'hue': hue, 'saturation': sat, 'value': val})
    

#%% 
### Launch app (Tkinter interface)
#%% 
root = Tk()
root.geometry("1450x950")
root.title('SC Robotics Mineral Library Builder')
root.update()
width= root.winfo_width()
height=root.winfo_height()
counter = 0
hfltr = [0,100]
vfltr = [0,100]
blur = 0
mineral_name= 'Unk'
fig = plt.figure()
x = IntVar
y = IntVar
pixel_x = []
pixel_y = []
df_current = pd.DataFrame(columns=['r', 'g', 'b', 'short', 'med', 'lon','L', 'A', 'B',
                        'hue', 'saturation', 'value','Mineral'])
left_panel =  Frame(root, width=width/2, height=height)
left_panel.grid(row=0, column=0)

right_panel =  Frame(root, width=width/2, height=height)
right_panel.grid(row=0, column=1)


lib_frame = Frame(right_panel, width=width/2, height=height/2, bg = 'white')
lib_frame.grid(row=0, column=0)

vals_frame = Frame(right_panel,width=width/2, height=height/2, bg = 'white')
vals_frame.grid(row=1, column=0)
    
im_frame = Frame(left_panel, width=width/2, height=height*3/6, bg='black')
im_frame.grid(row=0, column=0)

tool_frame = Frame(left_panel, width=width/2, height=height*2/6)
tool_frame.grid(row=1, column=0)

tags_frame = Frame(left_panel, width=width/2, height=height*1/6)
tags_frame.grid(row=2, column=0)

select_pixels_button = Button(tool_frame, text = "Exclude Pixel",relief='raised', command = select_pixels)
select_pixels_button.grid(row=0, column=1, padx=5, pady=5)

blur_slider = Scale(tool_frame, from_=0,to=10, orient=HORIZONTAL,
                fg='gray',bg = 'white',resolution=1)
blur_slider.grid(row=0, column=2,padx=5, pady=1)
blur_slider.set(blur)
blur_slider.bind("<ButtonRelease-1>",lambda e: change_blur())

#hue_slider = CTkRangeSlider(tool_frame,command = hue_filter, from_=0,to =100, orientation=HORIZONTAL,
 #               fg_color='gray',button_color = 'white', progress_color = 'white')
#hue_slider.grid(row=4, column=1, columnspan=2, padx=5,pady=5)
#hue_values=Label(tool_frame, text = f"Hue Quantile Filter:{min(hfltr): .2f} - {max(hfltr): .2f}")
#hue_values.grid(row=3, column=1, columnspan=2)

v_slider = CTkRangeSlider(tool_frame,command = v_filter, from_=0,to =100, orientation=HORIZONTAL,
                fg_color='gray',button_color = 'white', progress_color = 'white')
v_slider.grid(row=2, column=1, columnspan=2, padx=5,pady=5)
v_values=Label(tool_frame, text = f"Value Filter:{min(vfltr): .2f} - {max(vfltr): .2f}")
v_values.grid(row=1, column=1, columnspan=2)


lib_path = Button(tool_frame, text="From Existing Library",command= lambda: select_file('lib'))
lib_path.grid(row=0, column=0, padx=5, pady=1)

Label(tool_frame, text="OR").grid(row=1, column=0,padx=1, pady=5)

file_path = Button(tool_frame, text="Start New Library",command= lambda: select_file('new'))
file_path.grid(row=2, column=0, padx=5, pady=1)

check =  Button(tool_frame, text="Check", command= lambda: disp_current())
check.grid(row=0, column=3, padx=5, pady=5)

undo =  Button(tool_frame, text="Undo", command= lambda: undo_img())
undo.grid(row=1, column=3, padx=5, pady=5)

nxt =  Button(tool_frame, text="Next", command= lambda: next_img())
nxt.grid(row=2, column=3, padx=5, pady=5)

add = Button(tool_frame, text="Add to Library", command= lambda: add_img())
add.grid(row=0, column=4, padx=5, pady=5)

save = Button(tool_frame, text="Finish & Save Library", command=save_lib)
save.grid(row=1, column=4, padx=5, pady=5)

# kick off the GUI

root.mainloop()
#%% 
 
### Testing library 

#%% 
data = pd.read_csv('Libraries/Final/data_set.csv')
data.head()


### Looking at kNeighbors K values for best prediction

#%%
from sklearn import preprocessing
dt = data.copy()
#%%
tag = preprocessing.LabelEncoder()
tag.fit(dt['Mineral'])
dt['Mineral'] = tag.transform(dt['Mineral'])
#test_value = data.columns.to_list()
#test_value.remove('Mineral')
test_value = ['r','g','b']
value = dt[test_value]
mineral = dt['Mineral']
value_train, value_test, mineral_train, mineral_test = train_test_split(value,
                                                        mineral, test_size=0.3, random_state=5)
scaler=StandardScaler()
value_train_scaled=scaler.fit_transform(value_train)
value_test_scaled=scaler.fit_transform(value_test)
#%%
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
res=[]
sres = []
kval = [1,5,10,50,100,500]
for i in kval:
    clf = KNeighborsClassifier(n_neighbors=i)
    n = cross_validate(clf, value_train, mineral_train, cv=5, scoring='accuracy')
    ns = cross_validate(clf, value_train_scaled, mineral_train, cv=5, scoring='accuracy')
    res.append(n['test_score'].mean())
    sres.append(ns['test_score'].mean())
    
if max(res) >= max(sres):
    train_data = value_train
    k = kval[np.argmax(res)]
    print('not scaled')
if max(res) < max(sres):
    train_data = value_train_scaled
    k = kval[np.argmax(sres)]
    print('scaled')

#%%
plt.figure(figsize=(5, 5))
plt.plot(kval,res,linewidth=2.5)
plt.xscale("log")
plt.xlabel('neighbors')
plt.ylabel('Accuracmineral')
plt.show()

#%%

### Looking at prediction confidence by mineral type
#%%
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import classification_report
labres = pd.DataFrame()
kval = [5,10,50,100,500]
# Run through various values for k: use 5-fold cross validation, score with accuracmineral
for i in kval:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(value_train, mineral_train)
    prediction = clf.predict(value_test)
    f = classification_report(tag.inverse_transform(mineral_test), tag.inverse_transform(prediction),output_dict=True)
    labf = pd.DataFrame(f).transpose()
    labf['k'] = i
    labf['total accuracy'] =  labf.loc[['accuracy']].values[0,0]
    labres = pd.concat([labres,labf.drop(['accuracy','macro avg','weighted avg'])])
labres.reset_index(inplace=True)
labres = labres.rename(columns = {'index':'mineral'})
labres
#%%
import colorcet as cc
palette = sns.color_palette(cc.glasbey, n_colors=len(labres['mineral'].unique()))
sns.set_theme(style = "whitegrid", palette=palette)
g = sns.catplot(data=labres, x='k',y='f1-score',hue='mineral',kind='bar', dodge=True,
               height=7, aspect=1.5)
g.fig.get_axes()[0].set_ylim(0,1)
for container in g.fig.get_axes()[0].containers:
    g.fig.get_axes()[0].bar_label(container,size=10,fmt='%.2f')
g.fig.subplots_adjust(top=0.93)
g.fig.suptitle('F1 accuracy score of predictions with different k values by mineral type',fontsize=17)
plt.show()
#%%

### Random plots: Exploring library data set
#%%
fig = plt.figure(figsize=(8,4), dpi=100)
ax = fig.add_subplot()
palette = sns.color_palette(cc.glasbey, n_colors=len(data['Mineral'].unique()))
sns.set_theme(style = "ticks", palette=palette)
sns.histplot(data=data, x='hue', hue = 'Mineral', kde=True, fill=False, stat="density", ax= ax,element="step",legend=True)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(0.12,1),ncol = 2,frameon=False,title=None)
ax.set_xlim(0,1)
ax.set_title('Hue of Minerals in Library')
plt.show()
#%%
df = data.copy()
palette = sns.color_palette(cc.glasbey, n_colors=len(df['Mineral'].unique()))
sns.set_theme(style = "ticks", palette=palette)
sns.set_context("notebook", font_scale = 1.0)

width = 800
height=600
test_value = data.columns.to_list()
test_value.remove('Mineral')

if len(test_value) == 1:
    v=0
    x = str(test_value[int(v)])
    fig, ax = plt.subplots(figsize=(width/100,height/100), dpi=100)
    ax.set_title(f'Category Selected from Mineral Library')
    sns.histplot(data=df, x=x, hue = 'Mineral', kde=True, fill=False, stat="percent", ax=ax,
             element="step",legend= False,hue_order= df['Mineral'].unique())
    ax.set_xlim(0,max(df[x]))
    ax.set_xlabel(x)
if (len(test_value) <= 3) & (len(test_value) > 1):
    rows=len(test_value)
    fig, ax = plt.subplots(figsize=(width/100,height/100), dpi=100,nrows=rows, ncols=1,constrained_layout = True)
    v=0
    for i in range(0,rows):
        x = str(test_value[int(v)])
        ax[0].set_title(f'Category Selected from Mineral Library')
        sns.histplot(data=df, x=x, hue = 'Mineral', kde=True, fill=False, stat="percent", ax=ax[i],
                 element="step",legend= False,hue_order= df['Mineral'].unique())
        ax[i].set_xlim(0,max(df[x]))
        ax[i].set_xlabel(x)
        v = v+1
if len(test_value) > 3:
    n=len(test_value)
    rows= 3 - 1*(n==4)
    cols = int(np.ceil(n/rows))
    fig, ax = plt.subplots(figsize=(width/100,height/100), dpi=100,nrows=rows, ncols=cols,constrained_layout = True)
    v=0
    for i in range(0,rows):
        for j in range(0,cols):
            x = str(test_value[int(v)])
            fig.suptitle('Category Selected from Mineral Library')
            sns.histplot(data=df, x=x, hue = 'Mineral', kde=True, fill=False, stat="density", ax=ax[i,j],
                     element="step",legend=False,hue_order= df['Mineral'].unique())
            ax[i,j].set_xlabel(x)
            v = v+1
plt.show()
#%%
sns.set_theme(style = "ticks", palette=palette)
sns.set_context("notebook", font_scale = 0.65)
g = sns.pairplot(data,hue = "Mineral", vars = ["r","g","b"], size = 5, plot_kws=dict(marker="+", linewidth=1,alpha=0.1)
                 ,corner=True)


#%%
g = sns.pairplot(data,hue = "Mineral", vars = ["hue","saturation","value"], size = 5, plot_kws=dict(marker="+", linewidth=1,alpha=0.1)
                 ,corner=True)


#%%
g = sns.FacetGrid(data=data, col='Mineral',col_wrap=3, height=5, aspect= 1.5)
g.map(sns.histplot, 'L', color = 'black', fill=False, element="poly", stat="density", bins = 100)
g.map(sns.histplot, 'A', color = 'magenta', fill=False, element="poly", stat="density", bins = 100)
g.map(sns.histplot, 'B', color = 'blue', fill=False, element="poly", stat="density", bins = 100)
g.set(xlabel = "Intensity", ylabel = "Count Density")
#%%
g = sns.FacetGrid(data=data, col='Mineral',col_wrap=3, height=5, aspect= 1.5)
g.map(sns.histplot, 'short', color = 'red', fill=False, element="poly", stat="density", bins = 100)
g.map(sns.histplot, 'med', color = 'green', fill=False, element="poly", stat="density", bins = 100)
g.map(sns.histplot, 'lon', color = 'blue', fill=False, element="poly", stat="density", bins = 100)
g.set(xlabel = "Intensity", ylabel = "Count Density")
#%%
g = sns.FacetGrid(data=data, col='Mineral',col_wrap=3, height=5, aspect= 1.5)
g.map(sns.histplot, 'hue', color = 'red', fill=False, element="poly", stat="density", bins = 100)
g.map(sns.histplot, 'saturation', color = 'gray', fill=False, element="poly", stat="density", bins = 100)
g.map(sns.histplot, 'value', color = 'black', fill=False, element="poly", stat="density", bins = 100)
#%%
g = sns.FacetGrid(data=data, col='Mineral',col_wrap=3, height=5, aspect= 1.5)
g.map(sns.histplot, 'r', color = 'red', fill=False, element="poly", stat="density", bins = 100)
g.map(sns.histplot, 'g', color = 'green', fill=False, element="poly", stat="density", bins = 100)
g.map(sns.histplot, 'b', color = 'blue', fill=False, element="poly", stat="density", bins = 100)
g.set(xlabel = "Intensity", ylabel = "Count Density")
g.set(xlim=(0.005, 1), ylim=(0,10))
#%%
g = sns.jointplot(data=data, x='hue',y='value',  hue = 'Mineral', s = 4, marker="+",alpha = 0.1)
sns.move_legend(g.ax_joint, "lower right", bbox_to_anchor=(1.6,0.1), frameon=False)
plt.show()
#%%
