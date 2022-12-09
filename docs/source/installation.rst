Installation
============

Requirements
------------
It is known to work with Python 3.7.14 and the latest pandas & numpy versions for Python 3.7.14. As well as with any versions above that.

Furthermore it is known to NOT work with python 3.6.8 and pandas 1.1.5.

Preparation
-----------
pythonbody is not very well packaged, and dependencies are not very well tested. The simplest way of trying pythonbody is cloning the repo into the folder where your Jupyter notebook is or will be located.

.. code-block:: console

   $ git clone https://gitlab.com/shad0wfax/pythonbody.git

change into the pythonbody directory and install dependencies.

.. code-block:: console

   $ cd pythonbody 
   $ pip install -r requirements.txt

Building C-FFI-Libararies
-------------------------
For enhanced performance pythonbody uses C-FFI, which need to be build

.. code-block:: console

   move into ffi directory
   $ cd ffi

   Only required if you encounter errors later in the build process
   can be skipped on most systems
   $ autoreconf -f -i

   build ffi
   $ ./configure
   $ make clean && make

enjoy using pythonbody.
