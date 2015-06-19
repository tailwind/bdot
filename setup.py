
from distutils.core import Extension
from distutils.core import setup




# The minimum version of Cython required for generating extensions
min_cython_version = '0.20'

try:
    import Cython
    cur_cython_version = Cython.__version__
    from Cython.Distutils import build_ext
except:
    exit_with_error(
        "You need %(pkgname)s %(pkgver)s or greater to compile bcolz!"
        % {'pkgname': 'Cython', 'pkgver': min_cython_version})

if cur_cython_version < min_cython_version:
    exit_with_error(
        "At least Cython %s is needed so as to generate extensions!"
        % (min_cython_version))
else:
    print("* Found %(pkgname)s %(pkgver)s package installed."
          % {'pkgname': 'Cython', 'pkgver': cur_cython_version})

########### Project specific command line options ###########

class bdot_build_ext(build_ext):
    user_options = build_ext.user_options

    def initialize_options(self):
        self.from_templates = False
        build_ext.initialize_options(self)

    def run(self):

        build_ext.run(self)


######### End project specific command line options #########


VERSION = open('VERSION').read().strip()
# Create the version.py file
open('bdot/version.py', 'w').write('__version__ = "%s"\n' % VERSION)



# Sources & libraries
inc_dirs = ['bdot']
lib_dirs = []
libs = []
def_macros = []
sources_carray = ["bdot/carray_ext.pyx"]

# Include NumPy header dirs
from numpy.distutils.misc_util import get_numpy_include_dirs

inc_dirs.extend(get_numpy_include_dirs())

long_desc = """
Bdot does big dot products (by making your RAM bigger on the inside). 
It aims to let you do all the things you want to do with numpy, 
but don't have enough memory for. You can also use it to save space
in the cloud and quickly build production microservices.

It's pretty math.

Bdot is based on Bcolz and includes transparent disk-based storage. It was created 
to simplify production implementation of nearest neighbor search and other fancy 
machine learning rigmarole.
"""

setup(
    name = 'bdot',
    version = VERSION,
    description = 'Fast Matrix Multiply on Pretty Big Data (powered by Bcolz)',
    long_description= long_desc,
    url='https://github.com/tailwind/bdot',
    author='Waylon Flinn',
    cmdclass={'build_ext' : bdot_build_ext},
    ext_modules = [
          Extension("bdot.carray_ext",
                    include_dirs=inc_dirs,
                    sources=sources_carray)
          ],
    packages=['bdot', 'bdot.tests'],
    data_files=[('.' , ['VERSION'])],
    install_requires=['cython>=0.20', 'numpy>=1.7.0', 'bcolz>=0.9.0'],
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4'
    ],
    keywords='dot product matrix multiply compression bcolz numpy'
)
