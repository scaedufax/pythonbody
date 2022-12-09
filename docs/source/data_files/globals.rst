globals
=======

Load the output from `global.30` files

.. note::

   As `global` is a keyword in python, it has been renamed to `globals`
   in the entire project.

Use directly,

.. code-block:: python

   from pythonbody.data_files import globals
   globals_df = globals.load("/path/to/nbody/run")

or within the nbody class

.. code-block:: python

   from pythonbody import nbody
   n100k = nbody(data_path="/path/to/nbody/run",
                 nb_stdout_files="N100k.out")

   n100k.load() # Load all data_files
   n100k.load("globals") # Load globals data_file only

   n100k["globals"].keys() # show keys
   n100k.globals # alternative access
