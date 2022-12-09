stdout
======

Load the output from standard output of Nbody.

.. automodule:: pythonbody.data_files.stdout
   :members: load

.. note::

   - This contains a lot of data which is also found in `lagr.7` and many
     other files

Use directly,

.. code-block:: python

   from pythonbody.data_files import stdout
   stdout_df = stdout.load("/path/to/nbody/run/Nbody.out") #path to out file

or within the nbody class

.. code-block:: python

   from pythonbody import nbody
   n100k = nbody(data_path="/path/to/nbody/run",
                 nb_stdout_files="N100k.out")

   n100k.load() # Load all data_files
   n100k.load("stdout") # Load stdout data_file only

   n100k["stdout"].keys() # show keys
   n100k.stdout # alternative access
