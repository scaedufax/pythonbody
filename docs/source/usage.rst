Usage
=====

Unfortunatley this is not very well documented yet. See the examples folder for some examples.

Preparation
-----------

As pythonbody is not (yet) very well packaged, you might want simply append the
parent directory of pythonbody to your python PATH, so it can easily be used.

.. code-block:: python

   import sys
   sys.path.append("/path/to/pythonbody/parent/direcrotry")

Another option would be simply putting the pythonbody directory wherever you
want to use it. E.g. into the same directory as your jupyter notebook files.

nbody
-----
After your choice of preparation, the pythonbody package can be used.

Importing the :doc:`nbody`

.. code-block:: python

   from pythonbody import nbody

Initializing it

.. code-block:: python

   n100k = nbody(data_path="/path/to/nbody/run",
                 nb_stdout_files=["N100k.out"])

   # Load contents
   n100k.load()
