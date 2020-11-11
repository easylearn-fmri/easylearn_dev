import os
from eslearn.utils.el_call_powershell import PowerShell

if __name__ == '__main__':
    cmd = "pyuic5 -o ./easylearn_feature_engineering_gui.py ./easylearn_feature_engineering_gui.ui"
    with PowerShell('GBK') as ps:
        outs, errs = ps.run(cmd)
    print('error:', os.linesep, errs)
    print('output:', os.linesep, outs)

