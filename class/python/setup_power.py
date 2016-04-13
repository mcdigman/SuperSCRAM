from distutils.core import setup
from Cython.Build import cythonize
import numpy 
#from distutils.extension import Extension

#sourcefiles = ['class_interface.pyx']

#extensions = [Extension("iclass", sourcefiles)]

setup(
    ext_modules = cythonize('class_interface.pyx'),
    include_dirs=[numpy.get_include()]
)
