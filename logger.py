#-*- coding:utf-8 -*-
"""
Created on Fri Mar 29 15:32:38 2019

@author: DiXuanA
"""

import logging
import os
import win32com.client as win32
import warnings
import importlib
import sys
import pandas as pd
 
importlib.reload(sys)
warnings.filterwarnings('ignore')

class Logger(object):
    
    def __init__(self,logname):
        
        log_name = logname
        logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_name,
        filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        __content__ = pd.read_csv('receivers.csv')
        __receiver_list__ = [j for i in __content__.values for j in i]
        self.receivers= ''
        for i in __receiver_list__:
            self.receivers +=i
            self.receivers +=';'
            
        self.handler = logging.Handler() 
        
    def info(self, msg = " "):
        logging.info(str(msg))
        
    def warning(self, msg = " "):
        logging.warning(str(msg))
        
    def error(self, msg = " "):
        logging.error(str(msg))
        
    def critical(self, msg = " "):
        logging.critical(str(msg))
    
    def exception(self, msg = " "):
        logging.exception(str(msg))
        
    def sendemail(self,sub,body,face_img_path = 0):
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        print(self.receivers)
        mail.To = self.receivers
        mail.Subject = sub
        mail.Body = body
        
        file = os.getcwd()+face_img_path
        print(file)
        if face_img_path !=0:
            mail.Attachments.Add(r'{0}'.format(file))
        mail.Send()
        
    def close(self):
        self.handler.close()
