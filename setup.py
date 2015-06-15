
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

setup(
    name = 'bdot',
    version = VERSION,
    description = 'A fast dot product library built on Bcolz',
    long_description= 'Bdot makes your RAM bigger on the inside. It leverages the compression and chunking capabilities of Bcolz to compute dot products on data sets that don\'t fit in RAM. It can also (optionally) persist data sets to disk. It was created, initially, for nearest neighbor search.',
    url='https://github.com/pinleague/bdot',
    author='Waylon Flinn',
    cmdclass={'build_ext' : bdot_build_ext},
    ext_modules = [
          Extension("bdot.carray_ext",
                    include_dirs=inc_dirs,
                    sources=sources_carray)
          ],
    packages=['bdot', 'bdot.tests'],
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
    keywords='dot product compression bcolz numpy'
)
