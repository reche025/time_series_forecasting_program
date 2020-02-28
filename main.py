import os
import warnings
import itertools
import matplotlib
import seaborn as sns
import re
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

# import scipy.sparse.csgraph._validation    
# import scipy.sparse.linalg.dsolve.umfpack 
# import scipy.integrate.vode                
# import scipy.integrate.lsoda   
# import scipy.spatial.ckdtree             


from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from streamlit.ScriptRunner import StopException, RerunException

os.system("streamlit run --global.developmentMode=false forecasting_program_V1.py")