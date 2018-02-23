.. OTMF documentation master file

Welcome to OTMF's documentation!
==================================
 


* :ref:`genindex`
* :ref:`modindex`



This is the index page of the documentation of the OTMF package.

*Author* : Jean-Baptiste Courbot

Contact informations are on my website: http://www.jb-courbot.fr/

Licence
**************************************


The code provided is licensed under the CeCILL-B licence available in Licence_CeCILL-B.txt. The implications are the following :
"CeCILL-B follows the principle of the popular BSD license and its variants (Apache, X11 or W3C among others). In exchange for 
strong citation obligations (in all software incorporating a program covered by CeCILL-B and also through a Web site), the author
authorizes the reuse of its software without any other constraints." 

When using these codes, please cite :
Courbot, J. B., Monfrini, E., Mazet, V.  & Collet, C. (2018). Oriented Triplet Markov Fields. Pattern Recognition Letters, 103, 16-22.

The code is a prototypical implementation of Oriented Triplet Markov Field (OTMF) model introduced in this paper.

Installation
**************************************

To install this package, please run ::

    python setup.py install

in the main package folder.

There is no specific package requirement, most of the code uses numpy and scipy modules.

Once installed, all package components are available using::

    from otmf import *

*Note:* the code has been developed under Python 2.7, upon minor changes it should be able to run with Python 3.x.
        

Running the code
**************************************

Once the installation is made, a demontration jupyter notebook "demu.ipynb" is available.
This file should be self-explanatory with regards to the input parameters and the methods from the paper.

Documentation
**************************************
The index of individual modules and functons is available below.

Updates and contributions
**************************************

The proposed codes are, with respect to the paper, in their most advanced form.
If you spot any bug, feel free to contact me (website above).
Any contribution on model & algorithms implementations are of course welcome.



That's all folks !


.. toctree::
   :maxdepth: 3






