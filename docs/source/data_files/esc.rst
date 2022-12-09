esc
===

Load the output from `esc.11` files, containing data about escaping stars.

Use directly,

.. code-block:: python

   from pythonbody.data_files import esc
   esc_df = esc.load("/path/to/nbody/run")

or within the nbody class

.. code-block:: python

   from pythonbody import nbody
   n100k = nbody(data_path="/path/to/nbody/run",
                 nb_stdout_files="N100k.out")

   n100k.load() # Load all data_files
   n100k.load("esc") # Load esc data_file only

   n100k["esc"].keys() # show keys
   n100k.esc # alternative access
