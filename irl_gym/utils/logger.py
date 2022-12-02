"""
This module contains a simple logger

Syntax convention note: 
- Using leading "_" for function arguments (except gym standard vars)
- Using trailing "_" for member variables (except gym standard vars)
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"


class Logger():
    """   
    Simple logger with 5 settings:
    
    - **0**: off (default) 
    - **1**: debug
    - **2**: info
    - **3**: warn
    - **4**: error
    - **5**: fatal
    
    :param _log_level: (int) minimum log level, *default* 0
    """
    def __init__(self, _log_level = 0):
        super(Logger,self).__init__()
        
        self.ll_ = _log_level()
        levels = {1:"DEBUG: ", 2:"INFO: ", 3:"WARN: ", 4:"ERROR: ", 5:"FATAL: "}
        
        if self.ll_:
            self.l_str_ = levels[self.ll_]
            
    def print(self, _str, _level = 0):
        """
        Prints desired string if _level is greater than log_level

        :param _str: (str) log string
        :param _level: (int) level of log string, *default*: 0
        """
        if self.ll_ and _level >= self.ll_:
            print(self.l_str_ + _str)
            
    
        
