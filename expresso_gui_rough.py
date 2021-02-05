# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 17:11:07 2016

@author: Fruit Flies
"""

import matplotlib
matplotlib.use('TkAgg')

import os 
import sys 

if sys.version_info[0] < 3:
    from Tkinter import *
    import tkFileDialog
    from ttk import *
    import tkMessageBox
else:
    from tkinter import *
    from tkinter.ttk import *
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox

import h5py

import numpy as np
from scipy import interpolate

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
#import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
#from expresso_gui_params import guiParams

import csv
#------------------------------------------------------------------------------

class DirectoryFrame(Frame):
    """ Top UI frame containing the list of directories to be scanned. """

    def __init__(self, parent, col=0, row=0, filedir=None):
        Frame.__init__(self, parent)
                           
        self.lib_label = Label(parent, text='Directory list:') 
                               #foreground=guiParams['textcolor'], 
                               #background=guiParams['bgcolor'])
        self.lib_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)

        # listbox containing all selected additional directories to scan
        self.dirlist = Listbox(parent, width=64, height=8,
                               selectmode=SINGLE, exportselection=False)
                               #foreground=guiParams['textcolor'], 
                               #background=guiParams['listbgcolor'])
        self.dirlist.bind('<<ListboxSelect>>', self.on_select)
        self.dirlist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)

        self.btnframe = Frame(parent)
        self.btnframe.grid(column=col+2, row=row, sticky=NW)
        #self.btnframe.config(background=guiParams['bgcolor'])
        
        self.lib_addbutton =Button(self.btnframe, text='Add Directory',
                                   command= lambda: self.add_library(parent))
        self.lib_addbutton.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NW)
        
        self.lib_delbutton = Button(self.btnframe, text='Remove Directory',
                                  command=self.rm_library, state=DISABLED)
                                   
        self.lib_delbutton.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.dirlist.curselection():
            self.lib_delbutton.configure(state=NORMAL)
        else:
            self.lib_delbutton.configure(state=DISABLED)

    def add_library(self,parent):
        """ Insert every selected directory chosen from the dialog.
            Prevent duplicate directories by checking existing items. """

        dirlist = self.dirlist.get(0, END)
        newdir = Expresso.get_dir(parent)
        if newdir not in dirlist:
            self.dirlist.insert(END, newdir)

    def rm_library(self):
        """ Remove selected items from listbox when button in remove mode. """

        # Reverse sort the selected indexes to ensure all items are removed
        selected = sorted(self.dirlist.curselection(), reverse=True)
        for item in selected:
            self.dirlist.delete(item)

#------------------------------------------------------------------------------

class FileDataFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent)

        self.list_label = Label(parent, text='Detected files:')
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)

        self.filelistframe = Frame(parent) 
        self.filelistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.filelist = Listbox(self.filelistframe,  width=64, height=8, 
                                selectmode=SINGLE, exportselection=False)
        
        self.filelist.bind('<<ListboxSelect>>', self.on_select)
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.hscroll = Scrollbar(self.filelistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.filelistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.filelist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.filelist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.filelist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.filelist.yview)

        self.btnframe = Frame(parent)
        self.btnframe.grid(column=col+2, row=row, sticky=NW)
        
        # button used to initiate the scan of the above directories
        self.scan_btn =Button(self.btnframe, text='Get HDF5 Files',
                                        command= lambda: self.add_files(parent))
        #self.scan_btn['state'] = 'disabled'                                
        self.scan_btn.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NW)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_files
        self.remove_button.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)

        # label to show total files found and their size
        # this label is blank to hide it until required to be shown
        #self.total_label = Label(parent)
        #self.total_label.grid(column=col+1, row=row+2, padx=10, pady=2,
        #                      sticky=E)

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.filelist.curselection():
            self.remove_button.configure(state=NORMAL)
        else:
            self.remove_button.configure(state=DISABLED)
            
    def add_files(self,parent):
        newfiles = Expresso.scan_dirs(parent)
        
        if len(newfiles) > 0:
            self.filelist.delete(0,END)
            Expresso.clear_xplist(parent)
            Expresso.clear_channellist(parent)
            for file in tuple(newfiles):
                self.filelist.insert(END,file)
    
    def rm_files(self):
        selected = sorted(self.filelist.curselection(), reverse=True)
        for item in selected:
            self.filelist.delete(item)
              
        
#------------------------------------------------------------------------------
        
class XPDataFrame(Frame):        
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent)

        self.list_label = Label(parent, text='XP list:')
        self.list_label.grid(column=col+1, row=row, padx=10, pady=2, sticky=NW)

        self.xplistframe = Frame(parent) 
        self.xplistframe.grid(column=col+1, row=row+1, padx=10, pady=2, sticky=W)
        
        self.xplist = Listbox(self.xplistframe,  width=30, height=8, 
                              selectmode=SINGLE, exportselection=False)
        
        self.xplist.bind('<<ListboxSelect>>', self.on_select)
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.hscroll = Scrollbar(self.xplistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.xplistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.xplist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.xplist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.xplist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.xplist.yview)

        self.btnframe = Frame(parent)
        self.btnframe.grid(column=col+1, row=row+2, sticky=NW)
        
        # button used to initiate the scan of the above directories
        self.unpack_btn =Button(self.btnframe, text='Unpack HDF5 Files',
                                        command= lambda: self.add_xp(parent))
        self.unpack_btn.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NW)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_xp
        self.remove_button.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)

        # label to show total files found and their size
        # this label is blank to hide it until required to be shown
        #self.total_label = Label(parent)
        #self.total_label.grid(column=col+1, row=row+2, padx=10, pady=2,
        #                      sticky=E)

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.xplist.curselection():
            self.remove_button.configure(state=NORMAL)
        else:
            self.remove_button.configure(state=DISABLED)
            
    def add_xp(self,parent):
        self.xplist.delete(0,END)
        Expresso.clear_channellist(parent)
        newxp = Expresso.unpack_files(parent)
        
        for xp in tuple(newxp):
            self.xplist.insert(END,xp)
    
    def rm_xp(self):
        selected = sorted(self.xplist.curselection(), reverse=True)
        for item in selected:
            self.xplist.delete(item)
    """    
    def on_select(self, selection):
           Enable or disable based on a valid hdf5 selection. 
        if self.fdata_tree.curselection():
            self.unpack_btn.configure(state=NORMAL)
        else:
            self.unpack_btn.configure(state=DISABLED)
    """
#------------------------------------------------------------------------------
    
class ChannelDataFrame(Frame):        
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent)

        self.list_label = Label(parent, text='Channel list:')
        self.list_label.grid(column=col+1, row=row, padx=10, pady=2, sticky=N)
        
        self.channellistframe = Frame(parent) 
        self.channellistframe.grid(column=col+1, row=row+1, padx=10, pady=2, sticky=E)
        
        self.channellist = Listbox(self.channellistframe,  width=30, height=8,
                                   selectmode=SINGLE, exportselection=False)
        
        self.channellist.bind('<<ListboxSelect>>', self.on_select)
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.hscroll = Scrollbar(self.channellistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.channellistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.channellist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.channellist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.channellist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.channellist.yview)

        self.btnframe = Frame(parent)
        self.btnframe.grid(column=col+1, row=row+2, sticky=NE)
        
        # button used to initiate the scan of the above directories
        self.unpack_btn =Button(self.btnframe, text='Unpack XP',
                                        command= lambda: self.add_channels(parent))
        self.unpack_btn.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NE)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_channel
        self.remove_button.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NE)
                                
        self.selection_ind = []                        

        # label to show total files found and their size
        # this label is blank to hide it until required to be shown
        #self.total_label = Label(parent)
        #self.total_label.grid(column=col+1, row=row+2, padx=10, pady=2,
        #                      sticky=E)

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.channellist.curselection():
            self.remove_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.channellist.curselection(), reverse=True)
            #print(self.selection_ind)
        else:
            self.remove_button.configure(state=DISABLED)
            
    def add_channels(self,parent):
        self.channellist.delete(0,END)
        newchannels = Expresso.unpack_xp(parent)
        for channel in tuple(newchannels):
            self.channellist.insert(END,channel)
    
    def rm_channel(self):
        selected = sorted(self.channellist.curselection(), reverse=True)
        for item in selected:
            self.channellist.delete(item)
#------------------------------------------------------------------------------            
            
class FigureFrame(Frame):
    def __init__(self, parent,col=0,row=0):
        Frame.__init__(self, parent)
        
        #self.figure_label = Label(parent, text='Raw Data:')
        #self.figure_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)
       
        
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor = 'white')
        #self.fig = Figure()
        #self.subfig = self.fig.add_subplot(111)
        self.subplot0 = self.fig.add_subplot(211)
        self.subplot1 = self.fig.add_subplot(212, sharex=self.subplot0, sharey=self.subplot0)
        
        self.subplot0.set_ylabel('Liquid [nL]')
        self.subplot1.set_ylabel('Liquid [nL]')
       
        self.subplot1.set_xlabel('Time [units?]')
        
        self.subplot0.set_title('Raw Data')
        self.subplot1.set_title('Smoothed Data')
        #t = np.linspace(0.0, 3.0, 30)
        #s = np.sin(2*np.pi*t)
        
        #self.subplot0.plot(t, s)
        #self.subplot1.plot(t, 2*s)
        self.fig.set_tight_layout(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        #self.canvas.show()
        self.canvas.get_tk_widget().grid(column=col, row=row, columnspan = 4, rowspan = 6, padx=10, pady=2, sticky=E)
        #self.canvas.get_tk_widget().configure(background='red')
        self.canvas.show()
        
        self.toolbar_frame = Frame(parent)
        self.toolbar_frame.grid(column = col, row=row+6, columnspan = 2, sticky= W)        
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.pack()
        self.toolbar.update()
        
        
        self.btnframe = Frame(parent)
        self.btnframe.grid(column=col+2,row=row+6,columnspan = 2, sticky = E)
        
        self.plot_btn =Button(self.btnframe, text='Plot data',
                                        command= lambda: self.plot_data(parent))
        self.plot_btn.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NW)
        self.save_btn =Button(self.btnframe, text='Save results',
                                        command= lambda: self.save_results(parent))
        self.save_btn.grid(column=col+1, row=row, padx=10, pady=2,
                                sticky=NE)                        
        
        self.bouts = np.array([])
        
    def plot_data(self, parent):
        
        self.subplot0.cla()
        self.subplot1.cla()
        self.subplot0.set_ylabel('Liquid [nL]')
        self.subplot1.set_ylabel('Liquid [nL]')
        self.subplot1.set_xlabel('Time [units?]')
        self.subplot0.set_title('Raw Data')
        self.subplot1.set_title('Smoothed Data')
        
        dset, frames = Expresso.plot_channel(parent)
        
        if dset.size != 0:   
            dset_smooth, bouts, volumes = bout_analysis(dset,frames)
            
            self.bouts = bouts
            self.dset_smooth = dset_smooth
            
            self.subplot0.plot(np.squeeze(frames), np.squeeze(dset))
            self.subplot1.plot(frames, dset_smooth)
            
            for i in np.arange(bouts.shape[1]):
                self.subplot1.plot(frames[bouts[0,i]:bouts[1,i]], dset_smooth[bouts[0,i]:bouts[1,i]],'r-')
            
            self.subplot0.set_xlim([0,dset.size])
            self.subplot0.set_ylim([np.amin(dset),np.amax(dset)])
            self.fig.canvas.draw()
        else:
            tkMessageBox.showinfo(title='Error',
                                message='Invalid channel selection--no data in channel')
                                
    def save_results(self,parent):
        if sys.version_info[0] < 3:
            save_file = tkFileDialog.asksaveasfile(mode='wb', 
                            defaultextension=".csv")
            save_writer = csv.writer(save_file)
        else:
            save_filename = tkFileDialog.asksaveasfilename(defaultextension=".csv")
            save_file = open(save_filename, 'w', newline='')
            save_writer = csv.writer(save_file)
            
        bouts_transpose = np.transpose(self.bouts)
        if bouts_transpose.size > 0 :
            save_writer.writerow(['Bout Number'] + ['Bout Start'] + ['Bout End'])
            cc = 1            
            for row in bouts_transpose:
                new_row = np.insert(row,0,cc)
                save_writer.writerow(new_row)
                cc += 1
            
                                
            
#------------------------------------------------------------------------------

class Expresso(Tk):
    """The GUI and functions."""
    def __init__(self):
        Tk.__init__(self)
        
        # initialize important fields for retaining where we are in data space
        init_dirs = []
        self.initdirs = init_dirs
        
        datapath = []
        self.datadir_curr = datapath 
        
        filename = []
        self.filename_curr = filename 
        
        filekeyname = []
        self.filekeyname_curr = filekeyname
        
        grpnum = []
        self.grpnum_curr = grpnum 
        
        grp = []
        self.grp_curr = grp
        
        # run gui presets. may be unecessary
        self.init_gui()
        
        # initialize instances of frames created above
        self.dirframe = DirectoryFrame(self, col=0, row=0)
        self.fdata_frame = FileDataFrame(self, col=0, row=1)
        self.xpdata_frame = XPDataFrame(self, col=0, row=3)
        self.channeldata_frame = ChannelDataFrame(self, col=0, row=3)
        self.rawdata_plot = FigureFrame(self, col=4, row=0)
        
        for datadir in self.initdirs:
            self.dirframe.dirlist.insert(END, datadir)
        
        self.make_topmost()
        self.protocol("WM_DELETE_WINDOW", self.on_quit)
        
    def on_quit(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Quit","Do you want to quit?"):
            self.destroy()
            self.quit()
    
    def make_topmost(self):
        """Makes this window the topmost window"""
        self.lift()
        self.attributes("-topmost", 1)
        self.attributes("-topmost", 0) 
        
    def init_gui(self):
        """Label for GUI"""
        self.title('Expresso Data Analysis (rough version)')
        
        
        """ Menu bar """
        self.option_add('*tearOff', 'FALSE')
        self.menubar = Menu(self)
 
        self.menu_file = Menu(self.menubar)
        self.menu_file.add_command(label='Exit', command=self.on_quit)
 
        self.menu_edit = Menu(self.menubar)
 
        self.menubar.add_cascade(menu=self.menu_file, label='File')
        self.menubar.add_cascade(menu=self.menu_edit, label='Edit')
 
        self.config(menu=self.menubar)
        #self.configure(background='dim gray')
        #self.tk_setPalette(background=guiParams['bgcolor'],
        #                   foreground=guiParams['textcolor']) 
                           
        
    @staticmethod
    def get_dir(self):
        """ Method to return the directory selected by the user which should
            be scanned by the application. """

        # get user specified directory and normalize path
        seldir = tkFileDialog.askdirectory(initialdir=sys.path[0])
        if seldir:
            seldir = os.path.abspath(seldir)
            self.datadir_curr = seldir
            return seldir
    
    @staticmethod        
    def scan_dirs(self):
        # build list of detected files from selected paths
        files = [] 
        temp_dirlist = list(self.dirframe.dirlist.get(0, END))
        selected_ind = sorted(self.dirframe.dirlist.curselection(), reverse=True)
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select directory from which to grab hdf5 files')
            return files                    
        # print(temp_dirlist)
        temp_dir = temp_dirlist[selected_ind[0]]
        for file in os.listdir(temp_dir):
            if file.endswith(".hdf5"):
                files.append(file)
                    
        self.datadir_curr = temp_dir
        
        if len(files) > 0:
            return files
        else:
            tkMessageBox.showinfo(title='Error',
                                message='No HDF5 files found.')
            files = []
            return files                    
            
    @staticmethod 
    def unpack_files(self):
        selected_ind = sorted(self.fdata_frame.filelist.curselection(), reverse=True)
        #print(selected_ind)
        selected = [] 
        for ind in selected_ind: 
            selected.append(self.fdata_frame.filelist.get(ind))
        
        temp_dirlist = list(self.dirframe.dirlist.get(0, END))
        for dir in temp_dirlist:
            filename = os.path.join(dir,selected[0])
            #print(filename)
            if os.path.isfile(filename):
                self.filename_curr = filename 
                f = h5py.File(filename,'r')
                fileKeyNames = list(f.keys())
                return fileKeyNames    
    @staticmethod 
    def unpack_xp(self):
        selected_ind = sorted(self.xpdata_frame.xplist.curselection(), reverse=True)
        f = h5py.File(self.filename_curr,'r')
        fileKeyNames = list(f.keys())
        #print(type(selected_ind))
        
        grp = f.require_group(fileKeyNames[selected_ind[0]])
        groupKeyNames = list(grp.keys())
        if groupKeyNames:
            self.filekeyname_curr = fileKeyNames[selected_ind[0]]
            self.grp_curr = grp 
            self.grpnum_curr = selected_ind[0]
            return groupKeyNames
    
    @staticmethod
    def clear_xplist(self):
        self.xpdata_frame.xplist.delete(0,END)
    
    @staticmethod
    def clear_channellist(self):
        self.channeldata_frame.channellist.delete(0,END)
    
    #@staticmethod
    #def clear_plot(self):
        #in progress
          
    @staticmethod
    def plot_channel(self):
        dset = load_hdf5(self.filename_curr,self.grpnum_curr,self.channeldata_frame.selection_ind[0])        
        
        dset_check = (dset != -1)
        if (np.sum(dset_check) == 0):
            dset = np.array([])
            frames = np.array([])
            print('Problem with loading data - invalid data set')
            return (dset, frames)    
            
        frames = np.arange(0,dset.size)
        
        dset = dset[dset_check]
        frames = frames[np.squeeze(dset_check)]
        
        new_frames = np.arange(0,np.max(frames)+1)
        sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
        dset = sp_raw(new_frames)
        frames = new_frames
        
        try:
            return (dset, frames)
        except NameError:
            dset = np.array([])
            frames = np.array([])
            print('Problem with loading data - name error')
            return (dset, frames)
    
       
""" Run main loop """
if __name__ == '__main__':
    Expresso().mainloop()
    
"""    
    root = Tk()
    Expresso(root)
    root.mainloop()
"""
