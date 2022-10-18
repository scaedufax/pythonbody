# pythonbody
This project is currently under high development and might break with each update or commit. Furthermore as of now it is not very well documented.

## Requirements
It is known to work with Python 3.7.14 and the latest pandas & numpy versions for Python 3.7.14

## Installing pythonbody

### the local way

pythonbody is not very well packaged, and dependencies not very well tested. The simplest way of trying pythonbody is cloning the repo into the folder where your Jupyter notebook is located.

    git clone https://gitlab.com/shad0wfax/pythonbody.git

afterwards we need to build the ffi C-Libararies which pythonbody uses for enhanced performance

    # move into the ffi directory
    cd pythonbody/ffi

    # Only required if you encounter errors later in the build process
    # can be skipped on most systems
    autoreconf -f -i

    # build ffi
    ./configure
    make clean && make

last but not least install requirements

    pip install -r requirements.txt

enjoy using it.

## Using pythonbody

As already mentioned, pythonbody is not very well documented. have a look at the examples folder, that might give you a hints on how to use this package.
