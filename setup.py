from cx_Freeze import setup, Executable
import os

# os.environ['TCL_Library'] = r'C:\Program\Files\Python37\tcl\tcl8.6'
# os.environ['TK_Library'] = r'C:\Program\Files\Python37\tcl\tcl8.6'

build_exe_options = {
    'packages' : ['numpy', 'scipy'],
    'includes' : ['numpy', 'scipy']
}

executables = Executable('main.py', base = 'Win32GUI')


setup(name = 'Forecasting Program',
    version = '0.1',
    description = 'Forecasting time series data.',
    options = {
        'build_exe' : build_exe_options
    },
    executables = [executables]
)