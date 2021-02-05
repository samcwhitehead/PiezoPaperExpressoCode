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

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
#from matplotlib.figure import Figure

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
from batch_bout_analysis_func import batch_bout_analysis, save_batch_xlsx

#from PIL import ImageTk, Image
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
                               selectmode=EXTENDED, exportselection=False)
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
        
        self.lib_clearbutton = Button(self.btnframe, text='Clear All',
                                  command=self.clear_library)
                                   
        self.lib_clearbutton.grid(column=col, row=row+2, padx=10, pady=2,
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
    
    def clear_library(self):
        """ Remove all items from listbox when button pressed """
        self.dirlist.delete(0,END)        

#------------------------------------------------------------------------------

class FileDataFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent)

        self.list_label = Label(parent, text='Detected files:')
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)

        self.filelistframe = Frame(parent) 
        self.filelistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.filelist = Listbox(self.filelistframe,  width=64, height=8, 
                                selectmode=EXTENDED)
        
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
                                
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_files
        self.clear_button.grid(column=col, row=row+2, padx=10, pady=2,
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
        file_list = self.filelist.get(0,END)
        if len(newfiles) > 0:
            #file_list = self.filelist.get(0,END)
            #Expresso.clear_xplist(parent)
            #Expresso.clear_channellist(parent)
            for file in tuple(newfiles):
                if file not in file_list:
                    self.filelist.insert(END,file)
    
    def rm_files(self):
        selected = sorted(self.filelist.curselection(), reverse=True)
        for item in selected:
            self.filelist.delete(item)
    
    def clear_files(self):
        self.filelist.delete(0,END)        
              
        
#------------------------------------------------------------------------------
        
class XPDataFrame(Frame):        
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent)

        self.list_label = Label(parent, text='XP list:')
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)

        self.xplistframe = Frame(parent) 
        self.xplistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.xplist = Listbox(self.xplistframe,  width=64, height=8, 
                              selectmode=EXTENDED)
        
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
        self.btnframe.grid(column=col+2, row=row, sticky=NW)
        
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
         
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_xp
        self.clear_button.grid(column=col, row=row+2, padx=10, pady=2,
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
        xp_list = self.xplist.get(0,END)
        #Expresso.clear_channellist(parent)
        newxp = Expresso.unpack_files(parent)
        
        for xp in tuple(newxp):
            if xp not in xp_list:
                self.xplist.insert(END,xp)
    
    def rm_xp(self):
        selected = sorted(self.xplist.curselection(), reverse=True)
        for item in selected:
            self.xplist.delete(item)
        
    def clear_xp(self):
        self.xplist.delete(0,END)
    
#------------------------------------------------------------------------------
    
class ChannelDataFrame(Frame):        
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent)

        self.list_label = Label(parent, text='Channel list:')
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=N)
        
        self.channellistframe = Frame(parent) 
        self.channellistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=E)
        
        self.channellist = Listbox(self.channellistframe,  width=64, height=8,
                                   selectmode=EXTENDED)
        
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
        self.btnframe.grid(column=col+2, row=row, sticky=NW)
        
        # button used to initiate the scan of the above directories
        self.unpack_btn =Button(self.btnframe, text='Unpack XP',
                                        command= lambda: self.add_channels(parent))
        self.unpack_btn.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NW)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_channel
        self.remove_button.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
                   
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_channel
        self.clear_button.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=NW)                  
                                
        self.plot_button = Button(self.btnframe, text='Plot Channel')
        self.plot_button['state'] = 'disabled'
        self.plot_button['command'] = lambda: self.plot_channel(parent)
        self.plot_button.grid(column=col, row=row+3, padx=10, pady=2,
                                sticky=SW) 
        
        self.save_button = Button(self.btnframe, text='Save CSV')
        self.save_button['state'] = 'disabled'
        self.save_button['command'] = lambda: self.save_results(parent)
        self.save_button.grid(column=col, row=row+4, padx=10, pady=2,
                                sticky=SW)                        
                                
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
            self.plot_button.configure(state=NORMAL)
            self.save_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.channellist.curselection(), reverse=True)
            #print(self.selection_ind)
        else:
            self.remove_button.configure(state=DISABLED)
            self.plot_button.configure(state=DISABLED)
            self.save_button.configure(state=DISABLED)
            
            
    def add_channels(self,parent):
        channel_list = self.channellist.get(0,END)
        newchannels = Expresso.unpack_xp(parent)
        for channel in tuple(newchannels):
            if channel not in channel_list:
                self.channellist.insert(END,channel)
    
    def rm_channel(self):
        selected = sorted(self.channellist.curselection(), reverse=True)
        for item in selected:
            self.channellist.delete(item)
    
    def clear_channel(self):
        self.channellist.delete(0,END)        
    
    def plot_channel(self,parent):
        selected_ind = self.selection_ind
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select only one channel for plotting individual traces')
            return 
        
        channel_entry = self.channellist.get(selected_ind[0])
        dset, frames, t, dset_smooth, bouts, volumes = Expresso.get_channel_data(parent,channel_entry) 
        
        if dset.size != 0:   
            (dset,frames)
            self.bouts = bouts
            self.dset_smooth = dset_smooth
            self.volumes  = volumes
            self.t = t 
            #fig_window = Toplevel()
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, sharex=True, 
                                                sharey=True,figsize=(12, 7))
            
            self.ax1.set_ylabel('Liquid [nL]')
            self.ax2.set_ylabel('Liquid [nL]')
            self.ax2.set_xlabel('Time [s]')
            self.ax1.set_title('Raw Data')
            self.ax2.set_title('Smoothed Data')
            
            self.ax1.plot(t,dset)
            self.ax2.plot(t, dset_smooth)
            for i in np.arange(bouts.shape[1]):
                self.ax2.plot(t[bouts[0,i]:bouts[1,i]], dset_smooth[bouts[0,i]:bouts[1,i]],'r-')
                self.ax2.axvspan(t[bouts[0,i]],t[bouts[1,i]-1], 
                                 facecolor='grey', edgecolor='none', alpha=0.3)
                self.ax1.axvspan(t[bouts[0,i]],t[bouts[1,i]-1], 
                                 facecolor='grey', edgecolor='none', alpha=0.3)
                
            self.ax1.set_xlim([t[0],t[-1]])
            self.ax1.set_ylim([np.amin(dset),np.amax(dset)])    
                
            #self.fig.set_tight_layout(True)
               
            plt.subplots_adjust(bottom=0.2)
            self.ax_xrange = plt.axes([0.25, 0.1, 0.65, 0.03])
            self.ax_xmid = plt.axes([0.25, 0.06, 0.65, 0.03])

            self.slider_xrange = Slider(self.ax_xrange, 't range', 
                                        -1.0*np.amax(self.t), -1.0*3.0, 
                                        valinit=-1.0*np.amax(self.t))
            self.slider_xmid = Slider(self.ax_xmid, 't mid', 0.0, 
                                        np.amax(self.t), 
                                        valinit=np.amax(self.t)/2,
                                        facecolor='white')
            
            def update(val):
                xrange_val = -1.0*self.slider_xrange.val
                xmid_val = self.slider_xmid.val
                xmin = int(np.rint(np.amax([0, xmid_val - xrange_val/2])))
                xmax = int(np.rint(np.amin([self.dset_smooth.size, xmid_val + xrange_val/2])))
                xlim = [xmin, xmax]
                ymin = self.dset_smooth[xmin]+1
                ymax = self.dset_smooth[xmax-1]-1
                ylim = np.sort([ymin, ymax])
                self.ax2.set_xlim(xlim)
                self.ax2.set_ylim(ylim)
                #ax2_lim = ((xmin, np.amin(self.dset_smooth)), (xmax,np.amax(self.dset_smooth)))
                #self.ax2.update_datalim(ax2_lim)
                #self.ax2.autoscale()
                self.fig.canvas.draw_idle()
            self.slider_xrange.on_changed(update)
            self.slider_xmid.on_changed(update)
            
            
            full_channellist_entry = self.channellist.get(self.selection_ind[0])
            filepath, filekeyname, groupkeyname = full_channellist_entry.split(', ',2)
            dirpath, filename = os.path.split(filepath) 
            self.channel_name_full = filename + ", " + filekeyname + ", " + groupkeyname
            
            self.fig.canvas.set_window_title(self.channel_name_full)    
            
            #self.save_button['state'] = 'normal'
            #self.cursor = matplotlib.widgets.MultiCursor(self.fig.canvas, (self.ax1, self.ax2), 
            #                                        color='black', linewidth=1, 
            #                                        horizOn=False,vertOn=True)
            #plt.show()                                        
            #plt.show(self.fig)
            
        else:
            tkMessageBox.showinfo(title='Error',
                                message='Invalid channel selection--no data in channel')
    
    def save_results(self,parent):
        selected_ind = self.selection_ind
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select only one channel for plotting individual traces')
            return 
        
        full_channellist_entry = self.channellist.get(selected_ind[0])
        
        _, _, self.t, _, self.bouts, self.volumes = Expresso.get_channel_data(parent,full_channellist_entry)
        
        #full_channellist_entry = self.channellist.get(self.selection_ind[0])
        filepath, filekeyname, groupkeyname = full_channellist_entry.split(', ',2)
        dirpath, filename = os.path.split(filepath) 
        self.channel_name_full = filename + ", " + filekeyname + ", " + groupkeyname
        
        if self.bouts.size > 0 :
            bouts_transpose = np.transpose(self.bouts)
            volumes_col = self.volumes.reshape(self.volumes.size,1)
            row_mat = np.hstack((bouts_transpose,self.t[bouts_transpose],volumes_col))
            
            if sys.version_info[0] < 3:
                save_file = tkFileDialog.asksaveasfile(mode='wb', 
                                defaultextension=".csv")
                save_writer = csv.writer(save_file)
            else:
                save_filename = tkFileDialog.asksaveasfilename(defaultextension=".csv")
                save_file = open(save_filename, 'w', newline='')
                save_writer = csv.writer(save_file)
            
            save_writer.writerow([self.channel_name_full])
            save_writer.writerow(['Bout Number'] + ['Bout Start [idx]'] + \
                ['Bout End [idx]'] + ['Bout Start [s]'] + ['Bout End [s]']+ ['Volume [nL]'])
            cc = 1            
            for row in row_mat:
                new_row = np.insert(row,0,cc)
                save_writer.writerow(new_row)
                cc += 1
        else:
            tkMessageBox.showinfo(title='Error',
                                message='No feeding bouts to save')  
                          
#------------------------------------------------------------------------------

class BatchFrame(Frame):
    def __init__(self, parent,col=0,row=0):
        Frame.__init__(self, parent)
        
        self.list_label = Label(parent, text='Batch analyze list:')
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=S)
        
        self.batchlistframe = Frame(parent) 
        self.batchlistframe.grid(column=col, row=row+1, columnspan = 2, rowspan = 3, padx=10, pady=2, sticky=N)
        
        self.batchlist = Listbox(self.batchlistframe,  width=64, height=12,
                                   selectmode=EXTENDED)
        
        self.batchlist.bind('<<ListboxSelect>>', self.on_select)
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.hscroll = Scrollbar(self.batchlistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.batchlistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.batchlist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.batchlist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.batchlist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.batchlist.yview)
        
        self.entryframe = Frame(parent)
        self.entryframe.grid(column=col, row=row+2,columnspan = 3, rowspan = 2, pady=60, sticky=N)
        
        self.tmin_entry_label = Label(self.entryframe, text='t_min')
        self.tmin_entry_label.grid(column=col, row=0, padx=10, pady=2, sticky=N)
        self.tmin_entry = Entry(self.entryframe, width=8)
        self.tmin_entry.insert(END,'0')
        self.tmin_entry.grid(column=col, row=1,padx=10, pady=2, sticky=S)
        
        self.tmax_entry_label = Label(self.entryframe, text='t_max')
        self.tmax_entry_label.grid(column=col+1, row=0, padx=10, pady=2, sticky=N)
        self.tmax_entry = Entry(self.entryframe, width=8)
        self.tmax_entry.insert(END,'2000')
        self.tmax_entry.grid(column=col+1, row=1,padx=10, pady=2, sticky=S)
        
        self.tbin_entry_label = Label(self.entryframe, text='t_bin')
        self.tbin_entry_label.grid(column=col+2, row=0, padx=10, pady=2, sticky=N)
        self.tbin_entry = Entry(self.entryframe, width=8)
        self.tbin_entry.insert(END,'20')
        self.tbin_entry.grid(column=col+2, row=1,padx=10, pady=2, sticky=S)                           
        
        
        self.btnframe = Frame(parent)
        self.btnframe.grid(column=col, row=row+3, columnspan = 2, rowspan = 2, sticky=N)
        
        # button used to initiate the scan of the above directories
        self.add_btn =Button(self.btnframe, text='Add Channel(s) to Batch',
                                        command= lambda: self.add_to_batch(parent))
        self.add_btn.grid(column=col, row=row, padx=10, pady=2,
                                sticky=N)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Selected')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_channel
        self.remove_button.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=N)
        
        # button used to clear all batch files
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_batch
        self.clear_button.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=N)
                                
        self.plot_button = Button(self.btnframe, text='Plot Batch Analysis')
        #self.plot_button['state'] = 'disabled'
        self.plot_button['command'] = self.plot_batch
        self.plot_button.grid(column=col+1, row=row, padx=10, pady=2,
                                sticky=S) 
        
        self.save_button = Button(self.btnframe, text='Save Batch Results')
        #self.save_button['state'] = 'disabled'
        self.save_button['command'] = self.save_batch
        self.save_button.grid(column=col+1, row=row+1, padx=10, pady=2,
                                sticky=S)                        
        
        self.save_ts_button = Button(self.btnframe, text='Save Time Series')
        #self.save_button['state'] = 'disabled'
        self.save_ts_button['command'] = lambda: self.save_time_series(parent)
        self.save_ts_button.grid(column=col+1, row=row+2, padx=10, pady=2,
                                sticky=S)         
                                
        self.selection_ind = []                        

        # label to show total files found and their size
        # this label is blank to hide it until required to be shown
        #self.total_label = Label(parent)
        #self.total_label.grid(column=col+1, row=row+2, padx=10, pady=2,
        #                      sticky=E)

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.batchlist.curselection():
            self.remove_button.configure(state=NORMAL)
            #self.plot_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.batchlist.curselection(), reverse=True)
            #print(self.selection_ind)
        else:
            self.remove_button.configure(state=DISABLED)
            #self.plot_button.configure(state=DISABLED)
            
    def add_to_batch(self,parent):
        batch_list = self.batchlist.get(0,END)
        for_batch = Expresso.fetch_channels_for_batch(parent)
        for channel in tuple(for_batch):
            if channel not in batch_list:
                self.batchlist.insert(END,channel)
    
    def rm_channel(self):
        selected = sorted(self.batchlist.curselection(), reverse=True)
        for item in selected:
            self.batchlist.delete(item)
    
    def clear_batch(self):
        self.batchlist.delete(0,END)         
        
    def plot_batch(self):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            try:
                tmin = int(self.tmin_entry.get())
                tmax = int(self.tmax_entry.get())
                tbin = int(self.tbin_entry.get())
            except:
                tkMessageBox.showinfo(title='Error',
                                message='Set time range and bin size')
                return                
            
            (self.bouts_list, self.name_list, self.volumes_list, self.consumption_per_fly, 
             self.duration_per_fly, self.latency_per_fly, self.fig_raster, 
             self.fig_hist) = batch_bout_analysis(batch_list, tmin, tmax, tbin,True)
             
             #self.save_button['state'] = 'enabled'   
    
    def save_batch(self):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            try:
                tmin = int(self.tmin_entry.get())
                tmax = int(self.tmax_entry.get())
                tbin = int(self.tbin_entry.get())
            except:
                tkMessageBox.showinfo(title='Error',
                                message='Set time range and bin size')
                return                
            
            (self.bouts_list, self.name_list, self.volumes_list, self.consumption_per_fly, 
             self.duration_per_fly, self.latency_per_fly) = \
             batch_bout_analysis(batch_list, tmin, tmax, tbin,False)
            
            save_filename = tkFileDialog.asksaveasfilename(defaultextension=".xlsx")
            save_batch_xlsx(save_filename, self.bouts_list,self.name_list,
                        self.volumes_list,self.consumption_per_fly, 
                        self.duration_per_fly, self.latency_per_fly)

    def save_time_series(self,parent):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            save_dir = Expresso.get_dir(parent)
            
            for entry in batch_list:
                                
                dset, frames, t, dset_smooth, bouts, _ = Expresso.get_channel_data(parent,entry) 
                feeding_boolean = np.zeros([1,dset.size])
                for i in np.arange(bouts.shape[1]):
                    feeding_boolean[0,bouts[0,i]:bouts[1,i]] = 1
                row_mat = np.vstack((frames, t, dset, dset_smooth, feeding_boolean))
                row_mat = np.transpose(row_mat)
                
                filepath, filekeyname, groupkeyname = entry.split(', ',2)
                dirpath, filename = os.path.split(filepath) 
                save_name = filename[:-5] + "_" + filekeyname + "_" + groupkeyname + ".csv"
                save_path = os.path.join(save_dir,save_name)
                if sys.version_info[0] < 3:
                    out_path = open(save_path,mode='wb')
                else:
                    out_path = open(save_path, 'w', newline='')
                    
                save_writer = csv.writer(out_path)
                
                save_writer.writerow(['Idx'] + ['Time [s]'] + \
                    ['Data Raw [nL]'] + ['Data Smoothed [nL]'] + ['Feeding [bool]'])
                #cc = 1            
                for row in row_mat:
                    #new_row = np.insert(row,0,cc)
                    save_writer.writerow(row)
                    #cc += 1
                    
                out_path.close()
#------------------------------------------------------------------------------
#class LogoFrame(Frame):
#    def __init__(self, parent,col=0,row=0):
#        Frame.__init__(self, parent)
#        self.parent = parent
#        self.initLogo()
#    
#    def initLogo(self):
#        self.pack()
#        #insert logo
#        self.logo_canvas = Canvas(self.parent,width=1000,height=200)
#        self.logo_canvas.pack()
#        self.im = Image.open('C:\\Users\\Fruit Flies\\Documents\\Python Scripts\\Expresso GUI\\expresso_alpha.jpg')
#        self.logo_canvas.image = ImageTk.PhotoImage(self.im)
#        #self.photo = ImageTk.PhotoImage(self.im)
#        self.logo_canvas.create_image(0,0, image=self.logo_canvas.image, anchor=NW) 

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
        self.xpdata_frame = XPDataFrame(self, col=0, row=2)
        self.channeldata_frame = ChannelDataFrame(self, col=0, row=3)
        self.batchdata_frame = BatchFrame(self,col=3,row=0)
        #self.logo_frame = LogoFrame(self,col=0,row=0)
        
        for datadir in self.initdirs:
            self.dirframe.dirlist.insert(END, datadir)
        
        #self.rawdata_plot = FigureFrame(self, col=4, row=0)
        
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
        
        """ 
        parent.title('Expresso Data Analysis (rough version)')
        
        parent.option_add('*tearOff', 'FALSE')
        parent.menubar = Menu(parent)
 
        parent.menu_file = Menu(parent.menubar)
        parent.menu_file.add_command(label='Exit', command=self.on_quit)
 
        parent.menu_edit = Menu(parent.menubar)
 
        parent.menubar.add_cascade(menu=parent.menu_file, label='File')
        parent.menubar.add_cascade(menu=parent.menu_edit, label='Edit')
 
        parent.config(menu=parent.menubar)
        #self.configure(background='dim gray')
        #self.tk_setPalette(background=guiParams['bgcolor'],
        #                   foreground=guiParams['textcolor']) 
        """                   
        
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
        
        for ind in selected_ind:
            temp_dir = temp_dirlist[ind]
            for file in os.listdir(temp_dir):
                if file.endswith(".hdf5"):
                    files.append(os.path.join(temp_dir,file))
                    
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
        
        #temp_dirlist = list(self.dirframe.dirlist.get(0, END))
        #for dir in temp_dirlist:
        fileKeyNames = []
        for filename in selected:
            #filename = os.path.join(dir,selected[0])
            #print(filename)
            if os.path.isfile(filename):
                self.filename_curr = filename 
                f = h5py.File(filename,'r')
                for key in list(f.keys()):
                    if key.startswith('XP'):
                        fileKeyNames.append(filename + ", " + key)
        
        return fileKeyNames    
    
    @staticmethod 
    def unpack_xp(self):
        selected_ind = sorted(self.xpdata_frame.xplist.curselection(), reverse=True)
        groupKeyNames = []
        for ind in selected_ind:
            xp_entry = self.xpdata_frame.xplist.get(ind)
            filename, filekeyname = xp_entry.split(', ', 1)
            f = h5py.File(filename,'r')
            #fileKeyNames = list(f.keys())
            grp = f[filekeyname]
            for key in list(grp.keys()):
                groupKeyNames.append(filename + ', ' + filekeyname + ', ' + key) 
        #if groupKeyNames:
        #    self.filekeyname_curr = fileKeyNames[selected_ind[0]]
        #    self.grp_curr = grp 
        #    self.grpnum_curr = selected_ind[0]
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
    def get_channel_data(self,channel_entry):
        #selected_ind = self.channeldata_frame.selection_ind
        #if len(selected_ind) != 1:
        #    tkMessageBox.showinfo(title='Error',
        #                        message='Please select only one channel for plotting individual traces')
        #    return (dset,frames)
        #
        #channel_entry = self.channeldata_frame.channellist.get(selected_ind[0])
        filename, filekeyname, groupkeyname = channel_entry.split(', ',2)
        #print filename
        #print filekeyname
        #print groupkeyname
        dset, t = load_hdf5(filename,filekeyname,groupkeyname)        
        
        dset_check = (dset != -1)
        if (np.sum(dset_check) == 0):
            dset = np.array([])
            frames = np.array([])
            t = np.array([])
            dset_smooth = np.array([])
            bouts = np.array([])
            volumes = np.array([])
            print('Problem with loading data - invalid data set')
            return (dset, frames, t, dset_smooth, bouts, volumes)    
            
        frames = np.arange(0,dset.size)
        
        dset = dset[dset_check]
        frames = frames[np.squeeze(dset_check)]
        t = t[dset_check]
        
        new_frames = np.arange(0,np.max(frames)+1)
        sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
        sp_t = interpolate.InterpolatedUnivariateSpline(frames, t)
        dset = sp_raw(new_frames)
        t = sp_t(new_frames)
        frames = new_frames
        
        try:
            dset_smooth, bouts, volumes = bout_analysis(dset,frames)
            return (dset, frames, t, dset_smooth, bouts, volumes)
        except NameError:
            dset = np.array([])
            frames = np.array([])
            t = np.array([])
            dset_smooth = np.array([])
            bouts = np.array([])
            volumes = np.array([])
            print('Problem with loading data set--invalid name')
            return (dset, frames, t, dset_smooth, bouts, volumes)
    
    @staticmethod
    def fetch_channels_for_batch(self):
        selected_ind = self.channeldata_frame.selection_ind
        for_batch = []
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select channels to move to batch')
            return for_batch
        
        for ind in selected_ind: 
            for_batch.append(self.channeldata_frame.channellist.get(ind))
        
        return for_batch
    #@staticmethod
    #def get_batch_data(self):
#def main():
#    root = Tk()
#    root.geometry("300x280+300+300")
#    app = Expresso(root)
#    root.mainloop()
        
""" Run main loop """
if __name__ == '__main__':
    Expresso().mainloop()
    #main()
    
"""   
    root = Tk()
    Expresso(root)
    root.mainloop()
"""
