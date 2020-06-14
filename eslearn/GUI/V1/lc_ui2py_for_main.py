import os
from eslearn.utils.el_call_powershell import PowerShell

if __name__ == '__main__':
    cmd =  r'pyuic5 -o D:/My_Codes/easylearn-fmri/eslearn/GUI/easylearn_main_gui.py D:\My_Codes\easylearn-fmri\eslearn\GUI\easylearn_main_gui.ui'
    with PowerShell('GBK') as ps:
        outs, errs = ps.run(cmd)
    print('error:', os.linesep, errs)
    print('output:', os.linesep, outs)