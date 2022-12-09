lagr
====

Load the output from `lagr.7` files.

Use directly,

.. code-block:: python

   from pythonbody.data_files import lagr
   lagr_df = lagr.load("/path/to/nbody/run")

or within the nbody class

.. code-block:: python

   from pythonbody import nbody
   n100k = nbody(data_path="/path/to/nbody/run",
                 nb_stdout_files="N100k.out")

   n100k.load() # Load all data_files
   n100k.load("lagr") # Load globals data_file only

   n100k["lagr"].keys() # show keys
   n100k.lagr # alternative access
