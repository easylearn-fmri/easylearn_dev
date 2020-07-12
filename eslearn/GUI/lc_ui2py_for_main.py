import os
from eslearn.utils.el_call_powershell import PowerShell

if __name__ == '__main__':
    cmd =  r'pyuic5 -o ./easylearn_main_gui.py .\easylearn_main_gui.ui'
    with PowerShell('GBK') as ps:
        outs, errs = ps.run(cmd)
    print('error:', os.linesep, errs)
    print('output:', os.linesep, outs)