from distutils.core import setup, Extension
from Cython.Build import cythonize

module1 = Extension(name = 'cfxkmeans',
                    sources = ['src/cfxkmeans.pyx'], 
                    library_dirs=['lib/'],
                    include_dirs=['include/'],
                    extra_compile_args=['-fopenmp', '-lcfxkmeans','-std=c++11'], 
                    extra_link_args=   ['-fopenmp', '-lcfxkmeans','-std=c++11'])

setup ( ext_modules = cythonize([module1]),
        license = 'MIT')
